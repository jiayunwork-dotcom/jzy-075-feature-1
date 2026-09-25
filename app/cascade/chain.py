"""链路合成：段解析、等效系数卷积、稳定的链式求值。

段有两种形态：
- coefficients：调用方直接给出设计好的分子 / 分母系数；
- design：一个设计意图（类型 / 阶数 / 截止 / 窗），服务现场设计后再入链。

等效系数就是各段传递函数相乘：分子多项式顺序卷积、分母多项式顺序
卷积。卷积是精确运算，结果与单滤波器接口的系数格式完全一致，可以
直接喂回频响 / 零极点接口复核。

链式求值（chain_transfer）是整条链路的规范求值路径：逐段 Horner
求值再复数相乘。单段阶数低、系数形态好，误差在机器精度量级；而
把合成后的几十阶多项式直接代入单位圆，深阻带会因相消彻底失真，
因此内部计算绝不走展开式。
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass

from ..butterworth import design_iir_lowpass
from ..errors import FilterError
from ..fir import design_fir_lowpass
from ..polynomial import poly_multiply
from ..validation import validate_coefficients, validate_cutoff, validate_order
from ..windows import normalize_window_name

# 段描述 / 设计意图各自允许的字段，之外的字段一律拒绝（防止拼错被静默忽略）
_STAGE_FIELDS = {"name", "design", "coefficients"}
_DESIGN_FIELDS = {"type", "order", "cutoff", "window"}
_COEFFICIENT_FIELDS = {"b", "a"}


@dataclass(frozen=True)
class Stage:
    """链路中的一段：已解析出确定系数的最小单元。"""

    index: int  # 1 起始，与调用方给的顺序一致
    name: str
    source: str  # "design" 或 "coefficients"
    design: dict | None  # 设计意图段的规范化参数；系数段为 None
    b: list[float]
    a: list[float]

    @property
    def recursive(self) -> bool:
        """是否含非平凡分母（IIR 特征）；影响相位线性度的预期。"""
        return len(self.a) > 1


def _resolve_design(index: int, design: object) -> tuple[dict, list[float], list[float]]:
    """把设计意图现场落成系数；参数规则与单滤波器设计接口完全一致。"""
    if not isinstance(design, dict):
        raise FilterError(f"第 {index} 段：design 必须是对象")
    unknown = sorted(set(design) - _DESIGN_FIELDS)
    if unknown:
        raise FilterError(f"第 {index} 段：design 含未知字段 {unknown}")
    filter_type = design.get("type")
    if filter_type not in ("fir", "iir"):
        raise FilterError(f"第 {index} 段：design.type 只支持 fir / iir")
    if "order" not in design:
        raise FilterError(f"第 {index} 段：design 缺少 order（阶数）")
    if "cutoff" not in design:
        raise FilterError(f"第 {index} 段：design 缺少 cutoff（截止频率）")
    try:
        order = validate_order(design["order"])
        cutoff = validate_cutoff(design["cutoff"])
    except FilterError as exc:
        raise FilterError(f"第 {index} 段：{exc}") from exc

    if filter_type == "fir":
        window = design.get("window")
        if window is None:
            raise FilterError(f"第 {index} 段：FIR 设计必须通过 window 字段指定窗类型")
        try:
            window_name = normalize_window_name(window)
        except FilterError as exc:
            raise FilterError(f"第 {index} 段：{exc}") from exc
        coeffs = design_fir_lowpass(order, cutoff, window_name)
        summary = {"type": "fir", "order": order, "cutoff": cutoff, "window": window_name}
    else:
        if design.get("window") is not None:
            raise FilterError(f"第 {index} 段：IIR（巴特沃斯）设计不接受 window 参数")
        coeffs = design_iir_lowpass(order, cutoff)
        summary = {"type": "iir", "order": order, "cutoff": cutoff}
    return summary, coeffs["b"], coeffs["a"]


def _resolve_coefficients(index: int, coefficients: object) -> tuple[list[float], list[float]]:
    """系数段：复用统一的系数校验，错误信息前补上段号方便定位。"""
    if not isinstance(coefficients, dict):
        raise FilterError(f"第 {index} 段：coefficients 必须是对象")
    unknown = sorted(set(coefficients) - _COEFFICIENT_FIELDS)
    if unknown:
        raise FilterError(f"第 {index} 段：coefficients 含未知字段 {unknown}")
    if "b" not in coefficients or "a" not in coefficients:
        raise FilterError(f"第 {index} 段：coefficients 必须同时包含 b 与 a")
    try:
        b = validate_coefficients(coefficients["b"], "分子系数 b")
        a = validate_coefficients(coefficients["a"], "分母系数 a")
    except FilterError as exc:
        raise FilterError(f"第 {index} 段：{exc}") from exc
    return b, a


def resolve_stage(index: int, raw: object) -> Stage:
    """把调用方给的一段描述解析成确定系数的 Stage。"""
    if not isinstance(raw, dict):
        raise FilterError(f"第 {index} 段：段描述必须是对象")
    unknown = sorted(set(raw) - _STAGE_FIELDS)
    if unknown:
        raise FilterError(f"第 {index} 段：含未知字段 {unknown}")
    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        raise FilterError(f"第 {index} 段：name 必须是字符串")

    has_design = raw.get("design") is not None
    has_coefficients = raw.get("coefficients") is not None
    if has_design and has_coefficients:
        raise FilterError(f"第 {index} 段：design 与 coefficients 只能提供一个")
    if not has_design and not has_coefficients:
        raise FilterError(f"第 {index} 段：必须提供 design 或 coefficients 之一")

    if has_design:
        summary, b, a = _resolve_design(index, raw["design"])
        return Stage(index, name or f"段{index}", "design", summary, b, a)
    b, a = _resolve_coefficients(index, raw["coefficients"])
    return Stage(index, name or f"段{index}", "coefficients", None, b, a)


def equivalent_coefficients(stages: list[Stage]) -> tuple[list[float], list[float]]:
    """等效分子分母：各段系数顺序卷积（传递函数相乘）。

    单段链路是退化情形：原样返回该段系数，逐位一致，
    不因走了合成流程引入任何偏差。
    """
    b = list(stages[0].b)
    a = list(stages[0].a)
    for stage in stages[1:]:
        b = poly_multiply(b, stage.b)
        a = poly_multiply(a, stage.a)
    return b, a


def _eval_ascending(coeffs: list[float], z: complex) -> complex:
    """Horner 法求 c0 + c1 z^-1 + ... + cN z^-N 在给定 z 处的值。"""
    value = 0j
    for coefficient in reversed(coeffs):
        value = value / z + coefficient
    return value


def stage_transfer(b: list[float], a: list[float], z: complex) -> complex:
    """单段在 z 处的传输值 H(z) = B(z^-1) / A(z^-1)。"""
    denominator = _eval_ascending(a, z)
    if abs(denominator) < 1e-300:
        raise FilterError("分母多项式在该频点数值为 0，传输函数无定义")
    return _eval_ascending(b, z) / denominator


def chain_transfer(stages: list[Stage], frequency: float) -> complex:
    """整条链路在数字角频率 frequency（rad/采样）处的传输值。

    逐段求值再复数相乘——数学上等于等效传递函数，数值上远胜于
    对合成后的高阶多项式直接代入（后者在深阻带因相消失真）。
    """
    z = cmath.exp(1j * frequency)
    value = 1.0 + 0.0j
    for stage in stages:
        try:
            value *= stage_transfer(stage.b, stage.a, z)
        except FilterError as exc:
            raise FilterError(f"第 {stage.index} 段：{exc}") from exc
    return value
