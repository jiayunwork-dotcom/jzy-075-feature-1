"""把每段滤波器分解为一阶 / 二阶实系数节，按节组织整条链路。

为什么按节组织：链路合成后等效多项式动辄二三十阶，对这样的高阶
多项式直接求根，系数量级差异会把单位圆内的极点推到圆外（稳定性
误判）。每段本身阶数低、系数形态好，在段内求根再配成实系数节，
整条链路就是一串低阶节的连乘——极点位置与归因都保持精确。

节的定义：b0 + b1 z^-1 (+ b2 z^-2)，分母同理；实根单独成一阶节，
共轭根对成二阶节，z=0 处的根对应纯延迟节 z^-1。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis import plane_roots
from ..errors import FilterError
from ..polynomial import poly_multiply

# 共轭配对的容差：求根器输出的共轭对残差远小于此
_CONJUGATE_TOL = 1e-9
# |z| 小于该值视为原点处的根（纯延迟因子 z^-1）
_ORIGIN_TOL = 1e-12


@dataclass(frozen=True)
class Section:
    """一阶或二阶实系数节，系数按 z^-1 升幂。"""

    b: tuple[float, ...]
    a: tuple[float, ...]


def _pair_roots(roots: list[complex]) -> list[list[complex]]:
    """把实系数多项式的根配成实系数因子：实根单独、共轭对成对。"""
    remaining = sorted(roots, key=lambda z: (abs(z.imag), z.real, z.imag))
    factors: list[list[complex]] = []
    while remaining:
        root = remaining.pop(0)
        if abs(root.imag) <= _CONJUGATE_TOL:
            factors.append([complex(root.real, 0.0)])
            continue
        partner_index = None
        for i, other in enumerate(remaining):
            if (
                abs(other.real - root.real) <= _CONJUGATE_TOL
                and abs(other.imag + root.imag) <= _CONJUGATE_TOL
            ):
                partner_index = i
                break
        if partner_index is None:
            raise FilterError("求根结果缺少共轭配对，无法组成实系数节")
        factors.append([root, remaining.pop(partner_index)])
    return factors


def _real_coefficients(values: list[complex]) -> tuple[float, ...]:
    """节的系数必须是实数；虚部超出噪声量级说明配对出了问题。"""
    cleaned: list[float] = []
    for value in values:
        if abs(value.imag) > 1e-9 * max(1.0, abs(value.real)):
            raise FilterError("节的系数出现不可忽略的虚部，根配对失败")
        cleaned.append(float(value.real))
    return tuple(cleaned)


def sections_from_roots(
    zeros: list[complex], poles: list[complex], gain: float
) -> list[Section]:
    """由零点 / 极点与增益组织成一串一阶、二阶实系数节。

    原点处的根（来自 z^-k 纯延迟因子）单独成节 b = (0, 1)；
    增益挂到第一个非延迟节上，没有非延迟节时自成一节。
    """
    zero_factors: list[list[complex]] = []
    delay_count = 0
    for factor in _pair_roots(list(zeros)):
        if all(abs(r) < _ORIGIN_TOL for r in factor):
            delay_count += len(factor)
        else:
            zero_factors.append(factor)
    pole_factors = _pair_roots(list(poles))

    sections: list[Section] = [Section(b=(0.0, 1.0), a=(1.0,))] * delay_count
    count = max(len(zero_factors), len(pole_factors))
    gain_placed = False
    for i in range(count):
        b_poly: list[complex] = [1.0 + 0j]
        if i < len(zero_factors):
            for root in zero_factors[i]:
                b_poly = poly_multiply(b_poly, [1.0, -root])
        a_poly: list[complex] = [1.0 + 0j]
        if i < len(pole_factors):
            for root in pole_factors[i]:
                a_poly = poly_multiply(a_poly, [1.0, -root])
        b_coeffs = _real_coefficients(b_poly)
        if not gain_placed:
            # 增益挂到第一个非延迟节上
            b_coeffs = tuple(gain * c for c in b_coeffs)
            gain_placed = True
        sections.append(Section(b=b_coeffs, a=_real_coefficients(a_poly)))
    if not gain_placed:
        # 整条链路只有延迟节（或纯增益段），增益自成一节
        sections.append(Section(b=(float(gain),), a=(1.0,)))
    return sections


def stage_sections(b: list[float], a: list[float]) -> list[Section]:
    """把单段 (b, a) 分解为低阶节；增益取分子、分母首项非零系数之比。"""
    zeros, b_top = plane_roots(b)
    poles, a_top = plane_roots(a)
    return sections_from_roots(zeros, poles, b_top / a_top)


def stage_poles(a: list[float]) -> list[complex]:
    """单段分母在 z 平面的极点（段内求根，避免高阶合成多项式求根失真）。"""
    poles, _ = plane_roots(a)
    return poles
