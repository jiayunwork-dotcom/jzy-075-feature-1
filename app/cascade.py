"""级联链路：段解析、等效合成、稳健求值与稳定性归属（核心逻辑手写）。

一段链路要么是已落地的系数（b/a），要么是未落地的设计意图
（FIR 窗函数法 / IIR 巴特沃斯），本模块先把每一段解析成统一的
``ResolvedSegment``（b/a + 来源描述），再提供两条对外能力：

1. **等效系数**：按链路顺序把各段分子、分母分别做多项式卷积
   （``polynomial.poly_multiply``），得到与单滤波器接口同格式的
   ``{"b": [...], "a": [...]}``，可直接喂回频响 / 零极点接口对账。
2. **逐段求值与逐段求根**：高阶链路的频响 / 相位 / 指标计算不走
   二三十阶展开多项式——那会让分子里 (1+z^-1)^N 这类高重根在通用
   求根器里散掉（重组误差可达上百分贝），也会让展开系数的 Horner
   求值在深阻带发生灾难性抵消。改为：
   - 求值时每段（阶数 <= 16）各自 Horner 代入，幅度在 dB 域逐段
     累加（不连乘，避免 1e-300 以下下溢），相位逐段相加后统一展开；
     这与"每段单独求频响、再逐点相乘"是同一条数学路径。
   - 求极点时只对每段自己的（低阶）分母求根，极点因此可以精确
     归属到引入它的那一段，且不受其它段阶数牵连。

注意：级联顺序不改变等效传递函数，卷积结果与逐段响应只在浮点
舍入层面与顺序有关。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

from .analysis import _to_plane_roots
from .butterworth import design_iir_lowpass
from .errors import FilterError
from .fir import design_fir_lowpass
from .phase import unwrap_phase
from .polynomial import poly_multiply
from .validation import (
    validate_coefficients,
    validate_cutoff,
    validate_frequencies,
    validate_order,
)
from .windows import normalize_window_name

SEGMENT_KINDS = ("fir", "iir", "coefficients")

# 极点判定与 analysis 模块保持同一口径：根半径超出 1 + 容差才算圆外
STABILITY_TOLERANCE = 1e-9

_MIN_DB = -1000.0  # 与 analysis 模块一致的 dB 下限


@dataclass(frozen=True)
class ResolvedSegment:
    """一段已落地的滤波段。

    position 是它在链路里的 1-based 次序，仅用于对账与稳定性归属；
    spec 保存原始设计意图（阶数 / 截止 / 窗），系数段为空。
    """

    position: int
    kind: str
    b: list[float]
    a: list[float]
    spec: dict[str, object] = field(default_factory=dict)


def resolve_segments(raw_segments: object) -> list[ResolvedSegment]:
    """把请求里的段描述全部解析成 ResolvedSegment，非法输入抛 FilterError。"""
    if not isinstance(raw_segments, list):
        raise FilterError("级联链路的 segments 必须是数组")
    if len(raw_segments) == 0:
        raise FilterError("级联链路不能为空：至少需要一个滤波段")

    resolved: list[ResolvedSegment] = []
    for index, item in enumerate(raw_segments):
        position = index + 1
        if not isinstance(item, dict):
            raise FilterError(f"第 {position} 段必须是对象")
        kind = item.get("type")
        if not isinstance(kind, str):
            raise FilterError(
                f"第 {position} 段缺少 type，必须是 fir / iir / coefficients"
            )
        if kind not in SEGMENT_KINDS:
            raise FilterError(
                f"第 {position} 段类型 {kind!r} 不支持；必须是 fir / iir / coefficients"
            )

        spec: dict[str, object]
        if kind == "fir":
            if item.get("window") is None:
                raise FilterError(
                    f"第 {position} 段 FIR 设计必须通过 window 字段指定窗类型"
                )
            order = validate_order(item.get("order"))
            cutoff = validate_cutoff(item.get("cutoff"))
            window_name = normalize_window_name(item["window"])
            coefficients = design_fir_lowpass(order, cutoff, window_name)
            spec = {"order": order, "cutoff": cutoff, "window": window_name}
        elif kind == "iir":
            # 与单滤波器接口一致：IIR 不接受 window，避免参数被静默忽略
            if item.get("window") is not None:
                raise FilterError(
                    f"第 {position} 段 IIR（巴特沃斯）设计不接受 window 参数"
                )
            order = validate_order(item.get("order"))
            cutoff = validate_cutoff(item.get("cutoff"))
            coefficients = design_iir_lowpass(order, cutoff)
            spec = {"order": order, "cutoff": cutoff}
        else:
            b = validate_coefficients(item.get("b"), f"第 {position} 段分子系数 b")
            a = validate_coefficients(item.get("a"), f"第 {position} 段分母系数 a")
            coefficients = {"b": b, "a": a}
            spec = {}

        resolved.append(
            ResolvedSegment(
                position=position,
                kind=kind,
                b=coefficients["b"],
                a=coefficients["a"],
                spec=spec,
            )
        )
    return resolved


def cascade_coefficients(
    segments: list[ResolvedSegment],
) -> tuple[list[float], list[float]]:
    """按链路顺序卷积合成等效分子 / 分母系数。

    H_eq(z) = H_1(z) * ... * H_k(z)，落到系数上就是分子之间、
    分母之间分别做升幂多项式卷积。只有一段时原样返回，保证退化
    情形逐位相等，不经过任何算术改写。
    """
    if not segments:
        raise FilterError("级联链路不能为空：至少需要一个滤波段")
    if len(segments) == 1:
        return list(segments[0].b), list(segments[0].a)

    numerator: list[float] = [1.0]
    denominator: list[float] = [1.0]
    for segment in segments:
        numerator = poly_multiply(numerator, segment.b)
        denominator = poly_multiply(denominator, segment.a)
    return numerator, denominator


def _eval_ascending(coeffs: list[float], z: complex) -> complex:
    """Horner 求 c0 + c1 z^-1 + ... + cN z^-N（单段，阶数 <= 16，条件数可控）。"""
    value = 0j
    for coef in reversed(coeffs):
        value = value / z + coef
    return value


def segment_transfer(segment: ResolvedSegment, w: float) -> complex:
    """单段在角频率 w 处的复数传输值 H_k(e^{jw})。"""
    z = cmath.exp(1j * w)
    numerator = _eval_ascending(segment.b, z)
    denominator = _eval_ascending(segment.a, z)
    if abs(denominator) < 1e-300:
        raise ZeroDivisionError(f"第 {segment.position} 段分母在该频点数值为 0")
    return numerator / denominator


def chain_db_phase(
    segments: list[ResolvedSegment], w: float
) -> tuple[float, float]:
    """整条链路在角频率 w 处的 (幅度 dB, 主值相位 rad)。

    等价于先逐段求频响、再把各段复数响应相乘，但幅度改为 dB 域
    逐段累加（深阻带总响应可低于 1e-300，直接连乘会下溢），
    相位逐段主值相加（跨频点的展开交给 phase.unwrap_phase）。
    """
    total_db = 0.0
    total_phase = 0.0
    for segment in segments:
        value = segment_transfer(segment, w)
        amplitude = abs(value)
        total_db += (
            _MIN_DB if amplitude <= 1e-300 else 20.0 * math.log10(amplitude)
        )
        total_phase += math.atan2(value.imag, value.real)
    return total_db, total_phase


def chain_response_points(
    segments: list[ResolvedSegment], frequencies: list[float]
) -> list[tuple[float, float, float]]:
    """逐段路径下整条链路在各频点的 (角频率, 幅度 dB, 展开相位 rad)。"""
    if not segments:
        raise FilterError("级联链路不能为空：至少需要一个滤波段")
    freqs = validate_frequencies(frequencies)

    magnitudes: list[float] = []
    raw_phases: list[float] = []
    for w in freqs:
        db_value, phase = chain_db_phase(segments, w)
        magnitudes.append(db_value)
        raw_phases.append(phase)
    phases = unwrap_phase(raw_phases)
    return [
        (w, db_value, phase)
        for w, db_value, phase in zip(freqs, magnitudes, phases, strict=True)
    ]


def stability_report(segments: list[ResolvedSegment]) -> list[dict[str, object]]:
    """逐段对分母求根，列出所有单位圆外极点并归属到引入它的段。

    求根永远只在单段（低阶）分母上做：高阶展开分母系数动态范围
    悬殊，直接求根会把单位圆内极点错算到圆外；分段求根没有这个
    耦合，段号同时给出"是哪一段把链路弄不稳定的"。
    """
    report: list[dict[str, object]] = []
    for segment in segments:
        poles, _ = _to_plane_roots(segment.a)
        for pole in poles:
            radius = abs(pole)
            if radius > 1.0 + STABILITY_TOLERANCE:
                report.append(
                    {
                        "segment": segment.position,
                        "type": segment.kind,
                        "real": pole.real,
                        "imag": pole.imag,
                        "radius": radius,
                        "distance_to_unit_circle": radius - 1.0,
                    }
                )
    return report
