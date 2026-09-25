"""级联链路内部模块的单元测试。

重点盯住数值稳定性策略本身：
- 节分解能精确还原原段系数（含汉宁窗的纯延迟节）；
- 链式求值在通带与展开式一致（数学等价），在深阻带不随展开式失真；
- 对合成后的高阶分母直接求根会把圆内极点推到圆外——这条反面事实
  固定下来，说明链路模块为什么必须逐段求根、按节组织。
"""

from __future__ import annotations

import cmath
import math

import pytest

from app.cascade.chain import (
    chain_transfer,
    equivalent_coefficients,
    resolve_stage,
)
from app.cascade.schemas import CascadeRequest
from app.cascade.sections import stage_poles, stage_sections
from app.cascade.service import analyze_cascade
from app.polynomial import poly_multiply, roots_polynomial
from tests.conftest import linspace


def _horner_ascending(coeffs, z):
    """测试本地的 Horner 求值（与 analysis 模块同一格式）。"""
    value = 0j
    for coefficient in reversed(coeffs):
        value = value / z + coefficient
    return value


def _to_db(amplitude):
    return -1000.0 if amplitude <= 1e-300 else 20.0 * math.log10(amplitude)


def _tight_iir_chain():
    """四段 6 阶紧截止 IIR：24 阶链路，数值上最刁钻的情形。"""
    return [
        resolve_stage(i + 1, {"design": {"type": "iir", "order": 6, "cutoff": cutoff}})
        for i, cutoff in enumerate((0.05, 0.08, 0.1, 0.15))
    ]


# ------------------------------------------------------------ 节分解


@pytest.mark.parametrize(
    "raw",
    [
        {"design": {"type": "iir", "order": 5, "cutoff": 0.3}},
        {"design": {"type": "fir", "order": 10, "cutoff": 0.3, "window": "hann"}},
        {"coefficients": {"b": [1.0, -0.5, 0.25], "a": [1.0, -0.9, 0.8, -0.4]}},
        {"coefficients": {"b": [2.0], "a": [1.0]}},  # 纯增益段
    ],
)
def test_sections_reconstruct_stage_coefficients(raw):
    """每段拆成一阶 / 二阶节后，各节卷积必须还原原段系数。"""
    stage = resolve_stage(1, raw)
    sections = stage_sections(stage.b, stage.a)
    assert all(len(s.b) <= 3 and len(s.a) <= 3 for s in sections)  # 都是低阶节

    rebuilt_b = [1.0]
    rebuilt_a = [1.0]
    for section in sections:
        rebuilt_b = poly_multiply(rebuilt_b, list(section.b))
        rebuilt_a = poly_multiply(rebuilt_a, list(section.a))

    # 末端零系数对应无穷远处的根，不出现在节里；比较时两侧补齐到同长
    length_b = max(len(stage.b), len(rebuilt_b))
    padded_b = stage.b + [0.0] * (length_b - len(stage.b))
    padded_rebuilt_b = rebuilt_b + [0.0] * (length_b - len(rebuilt_b))
    length_a = max(len(stage.a), len(rebuilt_a))
    padded_a = stage.a + [0.0] * (length_a - len(stage.a))
    padded_rebuilt_a = rebuilt_a + [0.0] * (length_a - len(rebuilt_a))
    scale = max(1.0, max(abs(c) for c in padded_b))
    for expected, got in zip(padded_b, padded_rebuilt_b, strict=True):
        assert got == pytest.approx(expected, abs=1e-9 * scale)
    for expected, got in zip(padded_a, padded_rebuilt_a, strict=True):
        assert got == pytest.approx(expected, abs=1e-12)


def test_hann_fir_yields_pure_delay_section():
    """汉宁窗首系数为 0：节分解应给出一个纯延迟节 z^-1。"""
    stage = resolve_stage(
        1, {"design": {"type": "fir", "order": 10, "cutoff": 0.3, "window": "hann"}}
    )
    assert stage.b[0] == 0.0
    sections = stage_sections(stage.b, stage.a)
    delay_sections = [s for s in sections if s.b == (0.0, 1.0)]
    assert len(delay_sections) == 1


# ------------------------------------------------------- 链式求值


def test_chain_transfer_matches_expanded_form_in_passband():
    """通带内：逐段求值与展开等效多项式直接代入一致（数学等价的佐证）。"""
    stages = [
        resolve_stage(1, {"design": {"type": "iir", "order": 4, "cutoff": 0.2}}),
        resolve_stage(
            2, {"design": {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}}
        ),
    ]
    b_eq, a_eq = equivalent_coefficients(stages)
    for w in linspace(0.01 * math.pi, 0.3 * math.pi, 50):
        z = cmath.exp(1j * w)
        canonical = chain_transfer(stages, w)
        expanded = _horner_ascending(b_eq, z) / _horner_ascending(a_eq, z)
        assert abs(canonical - expanded) <= 1e-9 * abs(canonical), f"w={w}"


def test_chain_transfer_stays_accurate_where_expanded_form_diverges():
    """深阻带：展开式代入明显失真，链式求值仍给出真实深度。"""
    stages = _tight_iir_chain()
    b_eq, a_eq = equivalent_coefficients(stages)
    w = 0.9 * math.pi
    z = cmath.exp(1j * w)

    canonical_db = _to_db(abs(chain_transfer(stages, w)))
    expanded_db = _to_db(
        abs(_horner_ascending(b_eq, z) / _horner_ascending(a_eq, z))
    )
    # 真实响应在 -700 dB 以下；展开式在这里已偏离几十 dB
    assert canonical_db < -700.0
    assert abs(expanded_db - canonical_db) > 1.0


def test_expanded_denominator_rooting_would_misjudge_stability():
    """反面事实：对合成后的 24 阶分母直接求根，圆内极点被推到圆外。

    这条测试固定"为什么不能对高阶合成多项式直接求根"：同一条链路，
    展开式求根给出 |p|>1（误判不稳定），而链路模块逐段求根全部 |p|<1。
    """
    stages = _tight_iir_chain()
    _, a_eq = equivalent_coefficients(stages)
    assert len(a_eq) - 1 == 24

    expanded_roots = roots_polynomial(a_eq)
    assert max(abs(root) for root in expanded_roots) > 1.0  # 误判！

    for stage in stages:
        for pole in stage_poles(stage.a):
            assert abs(pole) < 1.0  # 逐段求根：全部在圆内


def test_chain_transfer_is_order_invariant():
    """链式求值与段的排列次序无关。"""
    raw_segments = [
        {"design": {"type": "iir", "order": 4, "cutoff": 0.2}},
        {"design": {"type": "fir", "order": 8, "cutoff": 0.4, "window": "hamming"}},
        {"coefficients": {"b": [1.0, -0.5], "a": [1.0, -0.8]}},
    ]
    forward = [resolve_stage(i + 1, raw) for i, raw in enumerate(raw_segments)]
    backward = [resolve_stage(i + 1, raw) for i, raw in enumerate(reversed(raw_segments))]
    for w in linspace(0.0, math.pi, 100):
        assert chain_transfer(forward, w) == pytest.approx(
            chain_transfer(backward, w), rel=1e-12, abs=0.0
        )


# ------------------------------------------------------- 服务层退化


def test_single_stage_service_result_is_bit_identical():
    """服务层退化情形：单段链路等效系数与输入逐位相等，且无指标。"""
    out = analyze_cascade(
        CascadeRequest(
            segments=[{"coefficients": {"b": [0.5, -0.25, 0.125], "a": [1.0, -0.9]}}]
        )
    )
    assert out["equivalent"]["b"] == [0.5, -0.25, 0.125]
    assert out["equivalent"]["a"] == [1.0, -0.9]
    assert out["equivalent"]["order"] == 2
    assert "metrics" not in out  # 未给频带则不计算指标
