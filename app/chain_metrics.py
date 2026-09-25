"""链路级指标：通带纹波、阻带衰减、相位线性度 / 群延时、稳定性汇总。

所有指标都基于合成后的整条链路计算（cascade 模块的逐段求值），
绝不对各段指标做简单加减——纹波与衰减在级联中会叠加或抵消，
只有整体求值才是对的。

求值按段（每段阶数 <= 16）各自 Horner 后在 dB / 相位域累加，
不在二三十阶展开多项式上做一次 Horner：后者在深阻带的分子
((1+z^-1)^N 乘积极化) 上会发生灾难性抵消，读数完全不可信。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .cascade import ResolvedSegment, chain_db_phase
from .errors import FilterError
from .phase import unwrap_phase

# 通带相位与最佳线性拟合的最大偏差超过该值即如实报告"非线性"
LINEAR_PHASE_TOLERANCE_RAD = 0.05

# 带内均匀采样点数；极值附近再做局部加密，保证纹波 / 衰减读数不丢峰
_BAND_SAMPLES = 1024
_REFINE_ROUNDS = 5


def validate_band(band: object, name: str) -> tuple[float, float]:
    """校验 [下限, 上限] 形式的归一化频带（相对奈奎斯特）。

    通带允许从 0 起，阻带允许到 1（奈奎斯特）止；两端必须有限、
    在 [0, 1] 内且严格递增，给反了或越界一律拒绝。
    """
    if not isinstance(band, Sequence) or isinstance(band, (str, bytes)):
        raise FilterError(f"{name} 必须是 [下限, 上限] 形式的数组")
    if len(band) != 2:
        raise FilterError(f"{name} 必须恰好包含下限、上限两个数值")
    cleaned: list[float] = []
    for index, item in enumerate(band):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise FilterError(f"{name}[{index}] 不是数值")
        value = float(item)
        if not math.isfinite(value):
            raise FilterError(f"{name}[{index}] 必须是有限数值")
        if not (0.0 <= value <= 1.0):
            raise FilterError(
                f"{name}[{index}]={value} 越界，归一化频率必须在 [0, 1] 内"
            )
        cleaned.append(value)
    if not cleaned[0] < cleaned[1]:
        raise FilterError(
            f"{name} 下限 {cleaned[0]} 必须严格小于上限 {cleaned[1]}（不能给反或相等）"
        )
    return cleaned[0], cleaned[1]


def _band_grid(lo_norm: float, hi_norm: float, count: int) -> list[float]:
    """归一化频带 [lo, hi]（相对奈奎斯特）上的均匀角频率网格。"""
    lo, hi = lo_norm * math.pi, hi_norm * math.pi
    step = (hi - lo) / (count - 1)
    return [lo + step * k for k in range(count)]


def _refine_extremum(
    evaluate: Callable[[float], float],
    grid: list[float],
    values: list[float],
    pick_max: bool,
) -> tuple[float, float]:
    """在均匀网格最优点两侧做局部加密，返回更准的 (频率, 极值)。

    纹波峰与阻带最坏点很少恰好落在网格点上，直接取网格最值会
    系统性偏乐观；每轮在当前最优点邻域内再均分采样，区间随之
    向最优点收缩，逐步逼近真实极值。
    """

    def better(candidate: float, incumbent: float) -> bool:
        return candidate > incumbent if pick_max else candidate < incumbent

    if pick_max:
        best_index = max(range(len(grid)), key=lambda k: values[k])
    else:
        best_index = min(range(len(grid)), key=lambda k: values[k])
    best_w, best_value = grid[best_index], values[best_index]

    # 局部加密的括号始终钳制在原频带内，最优点在带边时不会越界采样
    band_lo, band_hi = grid[0], grid[-1]
    lo = grid[best_index - 1] if best_index > 0 else grid[best_index]
    hi = grid[best_index + 1] if best_index < len(grid) - 1 else grid[best_index]
    for _ in range(_REFINE_ROUNDS):
        if hi <= lo:
            break
        step = (hi - lo) / 8.0
        for k in range(9):
            w = lo + step * k
            value = evaluate(w)
            if better(value, best_value):
                best_w, best_value = w, value
        lo = max(band_lo, best_w - step)
        hi = min(band_hi, best_w + step)
    return best_w, best_value


def _linear_fit_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """最小二乘拟合 y ≈ slope * x + intercept，返回 (slope, intercept)。"""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        raise FilterError("通带频点退化，无法拟合相位直线")
    slope = numerator / denominator
    return slope, mean_y - slope * mean_x


def compute_chain_metrics(
    segments: list[ResolvedSegment],
    passband: tuple[float, float],
    stopband: tuple[float, float],
    poles_report: list[dict[str, object]],
) -> dict[str, object]:
    """基于整条链路（逐段求值）计算链路级指标。

    参数
    ----
    segments:    已解析的链路各段（cascade.resolve_segments 的产物）。
    passband:    归一化通带 [lo, hi]，相对奈奎斯特。
    stopband:    归一化阻带 [lo, hi]，相对奈奎斯特。
    poles_report: 分段稳定性归属（圆外极点列表，带段号）。
    """
    pass_lo, pass_hi = passband
    stop_lo, stop_hi = stopband

    def chain_db(w: float) -> float:
        return chain_db_phase(segments, w)[0]

    # ---- 通带：参考电平取带内最大增益，纹波 = 最大起伏（峰峰值）----
    pass_grid = _band_grid(pass_lo, pass_hi, _BAND_SAMPLES)
    pass_db = [chain_db(w) for w in pass_grid]
    reference_w, reference_db = _refine_extremum(chain_db, pass_grid, pass_db, True)
    worst_pass_w, worst_pass_db = _refine_extremum(chain_db, pass_grid, pass_db, False)
    ripple_db = reference_db - worst_pass_db

    # ---- 阻带：最坏（最不衰减）点相对参考电平压下去多少 dB ----
    stop_grid = _band_grid(stop_lo, stop_hi, _BAND_SAMPLES)
    stop_db = [chain_db(w) for w in stop_grid]
    worst_stop_w, worst_stop_db = _refine_extremum(chain_db, stop_grid, stop_db, True)
    attenuation_db = reference_db - worst_stop_db

    # ---- 相位线性度与群延时：通带内展开相位对最佳直线的偏离 ----
    raw_phases = [chain_db_phase(segments, w)[1] for w in pass_grid]
    phases = unwrap_phase(raw_phases)
    slope, intercept = _linear_fit_slope(pass_grid, phases)
    residuals = [
        phase - (slope * w + intercept) for w, phase in zip(pass_grid, phases)
    ]
    max_deviation = max(abs(r) for r in residuals)
    rms_deviation = math.sqrt(sum(r * r for r in residuals) / len(residuals))

    group_delays = [
        -(phases[k + 1] - phases[k]) / (pass_grid[k + 1] - pass_grid[k])
        for k in range(len(pass_grid) - 1)
    ]
    group_delay_samples = sum(group_delays) / len(group_delays)
    group_delay_variation = max(group_delays) - min(group_delays)

    unstable_segments = sorted({int(p["segment"]) for p in poles_report})

    return {
        "reference_level_db": reference_db,
        "reference_frequency": reference_w / math.pi,
        "passband_ripple_db": ripple_db,
        "passband_max_db": reference_db,
        "passband_min_db": worst_pass_db,
        "passband_min_frequency": worst_pass_w / math.pi,
        "stopband_attenuation_db": attenuation_db,
        "stopband_worst_db": worst_stop_db,
        "stopband_worst_frequency": worst_stop_w / math.pi,
        "group_delay_samples": group_delay_samples,
        "group_delay_variation_samples": group_delay_variation,
        "phase_max_deviation_rad": max_deviation,
        "phase_rms_deviation_rad": rms_deviation,
        "linear_phase": max_deviation <= LINEAR_PHASE_TOLERANCE_RAD,
        "stable": not poles_report,
        "unstable_segments": unstable_segments,
        "unstable_poles": poles_report,
    }
