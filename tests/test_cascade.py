"""级联链路核心逻辑测试：合成、顺序无关、数值稳定、退化、衰减方向、非法输入。

六条关系都有对应自动化用例：
1. 等效系数频响 == 每段频响逐点复数相乘（验证卷积合成）；
2. 换段次序，等效频响完全一致；
3. 高阶链路（20 阶以上）逐段求根，稳定极点全部判在单位圆内；
   同时对照证明：同样的高阶展开分母直接求根会把稳定极点算到圆外，
   即数值守护这一层不是多余的；
4. 只有一段时等效结果与该段逐位相等；
5. 再串同类型低通，阻带最坏衰减只深不浅；
6. 空链路 / 非数值 / 全零分母 / 阶数截止越界 / 频带给反或越界一律拒绝。
"""

from __future__ import annotations

import cmath
import math

import pytest

from app.analysis import (
    _to_plane_roots,
    _transfer_from_coefficients,
    frequency_response,
    zero_pole_analysis,
)
from app.cascade import (
    cascade_coefficients,
    chain_db_phase,
    chain_response_points,
    resolve_segments,
    stability_report,
)
from app.chain_metrics import compute_chain_metrics, validate_band
from app.errors import FilterError
from app.polynomial import roots_polynomial
from tests.conftest import linspace


# ---------------------------------------------------------------------------
# 公共构造
# ---------------------------------------------------------------------------

def mixed_chain_payload():
    """低阶 IIR 压工频附近 + 高阶 FIR 削高频毛刺 + 一段已落地系数整形。"""
    return [
        {"type": "iir", "order": 4, "cutoff": 0.2},
        {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"},
        {"type": "coefficients", "b": [1.0, -0.5], "a": [1.0]},
    ]


def resolve(payload):
    return resolve_segments(payload)


# ---------------------------------------------------------------------------
# 关系一：等效系数频响 == 每段频响逐点相乘（dB 与相位都对）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_equivalent_coefficients_response_equals_per_segment_product(seed):
    payload = [
        {"type": "iir", "order": 4, "cutoff": 0.18 + 0.02 * seed},
        {"type": "fir", "order": 12, "cutoff": 0.30 + 0.02 * seed, "window": "hamming"},
        {"type": "coefficients", "b": [1.0, -0.3 - 0.05 * seed], "a": [1.0, 0.2]},
    ]
    segments = resolve(payload)
    b_eq, a_eq = cascade_coefficients(segments)

    # 合成阶数 = 各段阶数之和
    assert len(b_eq) - 1 == sum(len(s.b) - 1 for s in segments)
    assert len(a_eq) - 1 == sum(len(s.a) - 1 for s in segments)

    # 频点避开 w=pi：IIR 零点恰在 z=-1，该点两侧都是浮点噪声底，dB 比较无意义
    frequencies = linspace(0.0, 0.95 * math.pi, 61)
    points = frequency_response(b_eq, a_eq, frequencies)

    for w, point in zip(frequencies, points, strict=True):
        # 参照：每段单独求频响，再把各段复数响应逐点相乘
        z = cmath.exp(1j * w)
        product = 1.0 + 0j
        for segment in segments:
            product *= _transfer_from_coefficients(segment.b, segment.a, z)
        ref_db = 20.0 * math.log10(abs(product))
        ref_phase = math.atan2(product.imag, product.real)

        # 展开多项式 Horner 的浮点噪声是线性域绝对噪声，折算到 dB / 相位后
        # 随衰减深度近似平方放大（实测 -206 dB 处约 3e-8）；通带与过渡带
        # 内仍是 1e-9 量级的硬约束
        tol = 1e-9 * max(1.0, (abs(ref_db) / 10.0) ** 2)
        assert point.magnitude_db == pytest.approx(ref_db, abs=tol), (
            f"w={w:.4f}: 等效系数 {point.magnitude_db:.6f} dB 与逐段乘积 {ref_db:.6f} dB 不一致"
        )
        # 主值相位对齐后比较（频响接口给的是展开相位，减去最近的 2pi 倍数）
        wrapped = (
            (point.phase_rad - ref_phase + math.pi) % (2.0 * math.pi)
        ) - math.pi
        assert wrapped == pytest.approx(0.0, abs=tol)


def test_cascade_response_points_match_per_segment_product():
    """逐段求值路径（链路指标使用）与逐段复数相乘参照逐点一致。"""
    segments = resolve(mixed_chain_payload())
    # 避开 w=pi：z=-1 是 IIR 精确零点，幅度下溢后相位失去意义
    frequencies = linspace(0.0, 0.98 * math.pi, 81)
    points = chain_response_points(segments, frequencies)

    for w, (_freq, db_value, phase) in zip(frequencies, points, strict=True):
        z = cmath.exp(1j * w)
        product = 1.0 + 0j
        for segment in segments:
            product *= _transfer_from_coefficients(segment.b, segment.a, z)
        ref_db = 20.0 * math.log10(abs(product))
        assert db_value == pytest.approx(ref_db, abs=1e-9)
        wrapped = ((phase - math.atan2(product.imag, product.real) + math.pi)
                   % (2.0 * math.pi)) - math.pi
        assert wrapped == pytest.approx(0.0, abs=1e-9)


def test_equivalent_coefficients_feed_back_to_zero_poles():
    """等效系数可直接喂回现有零极点分析，段内 IIR 极点全部保留。"""
    segments = resolve(mixed_chain_payload())
    b_eq, a_eq = cascade_coefficients(segments)
    result = zero_pole_analysis(b_eq, a_eq)
    # 4 阶 IIR 贡献 4 个极点，其余段无极点
    assert len(result["poles"]) == 4
    assert result["stable"] is True


# ---------------------------------------------------------------------------
# 关系二：级联顺序无关
# ---------------------------------------------------------------------------

def test_cascade_response_independent_of_ordering():
    payload = mixed_chain_payload()
    forward = resolve(payload)
    reversed_chain = resolve(list(reversed(payload)))

    frequencies = linspace(0.0, math.pi, 101)
    db_forward = [chain_db_phase(forward, w)[0] for w in frequencies]
    db_reversed = [chain_db_phase(reversed_chain, w)[0] for w in frequencies]
    for w, x, y in zip(frequencies, db_forward, db_reversed, strict=True):
        assert x == pytest.approx(y, abs=1e-12), f"w={w:.4f} 处换序后频响不一致"

    phase_forward = [chain_db_phase(forward, w)[1] for w in frequencies]
    phase_reversed = [chain_db_phase(reversed_chain, w)[1] for w in frequencies]
    for x, y in zip(phase_forward, phase_reversed, strict=True):
        assert x == pytest.approx(y, abs=1e-12)


def test_cascade_coefficients_independent_of_ordering():
    """多项式卷积可交换，换序后的等效系数只差浮点舍入。"""
    payload = mixed_chain_payload()
    b1, a1 = cascade_coefficients(resolve(payload))
    b2, a2 = cascade_coefficients(resolve(list(reversed(payload))))
    assert len(b1) == len(b2)
    assert len(a1) == len(a2)
    scale_b = max(1.0, max(abs(v) for v in b1))
    scale_a = max(1.0, max(abs(v) for v in a1))
    assert max(abs(x - y) for x, y in zip(b1, b2)) < 1e-13 * scale_b
    assert max(abs(x - y) for x, y in zip(a1, a2)) < 1e-13 * scale_a


# ---------------------------------------------------------------------------
# 关系三：高阶链路的数值稳定性
# ---------------------------------------------------------------------------

def test_high_order_stable_chain_poles_all_inside_unit_circle():
    """三段 IIR 串到 38 阶：逐段求根，所有稳定极点仍判在圆内。"""
    segments = resolve([
        {"type": "iir", "order": 16, "cutoff": 0.10},
        {"type": "iir", "order": 16, "cutoff": 0.12},
        {"type": "iir", "order": 6, "cutoff": 0.20},
    ])
    assert sum(len(s.a) - 1 for s in segments) == 38

    report = stability_report(segments)
    assert report == []

    # 每段极点逐个核对：都在圆内，且最靠圆的极点也没有被推到圆外
    for segment in segments:
        poles, _ = _to_plane_roots(segment.a)
        assert poles
        for pole in poles:
            assert abs(pole) < 1.0 - 1e-6


def test_high_order_fir_chain_is_stable_and_evaluable():
    """四段 FIR 串到 40 阶：无极点、通带纹波与阻带读数有限可用。"""
    segments = resolve([
        {"type": "fir", "order": 12, "cutoff": 0.3, "window": "hamming"},
        {"type": "fir", "order": 10, "cutoff": 0.25, "window": "hann"},
        {"type": "fir", "order": 10, "cutoff": 0.28, "window": "hamming"},
        {"type": "fir", "order": 8, "cutoff": 0.32, "window": "rectangular"},
    ])
    b_eq, _ = cascade_coefficients(segments)
    assert len(b_eq) - 1 == 40
    assert stability_report(segments) == []

    metrics = compute_chain_metrics(
        segments, (0.0, 0.15), (0.45, 1.0), stability_report(segments)
    )
    assert metrics["stable"] is True
    assert math.isfinite(metrics["passband_ripple_db"])
    assert math.isfinite(metrics["stopband_attenuation_db"])
    assert metrics["stopband_attenuation_db"] > 40.0
    # 纯 FIR 链相位仍线性，群延时 = 各段 order/2 之和 = 6+5+5+4 = 20
    assert metrics["linear_phase"] is True
    assert metrics["group_delay_samples"] == pytest.approx(20.0, abs=1e-6)


def test_high_order_deep_stopband_response_matches_per_segment_reference():
    """38 阶链深阻带 -300 dB 量级：逐段路径与逐段乘积参照一致。

    对照路径：对 38 阶展开多项式直接 Horner / 直接求根都不可信——
    通带里展开求值会算出 -200 dB 以下的假读数，求根会把十余个
    单位圆内极点算到圆外。本用例把这两个对照事实也钉死，
    防止以后有人把"分段"简化回"一股脑卷完再算"。
    """
    segments = resolve([
        {"type": "iir", "order": 16, "cutoff": 0.10},
        {"type": "iir", "order": 16, "cutoff": 0.12},
        {"type": "iir", "order": 6, "cutoff": 0.20},
    ])
    b_eq, a_eq = cascade_coefficients(segments)

    # 深阻带 -322 dB：逐段路径必须对
    w = 0.3 * math.pi
    z = cmath.exp(1j * w)
    product = 1.0 + 0j
    for segment in segments:
        product *= _transfer_from_coefficients(segment.b, segment.a, z)
    ref_db = 20.0 * math.log10(abs(product))
    assert ref_db < -300.0  # 确认确实是深阻带场景

    db_value, _ = chain_db_phase(segments, w)
    assert db_value == pytest.approx(ref_db, abs=1e-9)

    # 同样是展开系数，这里已开始出现可测的误差，且比分段路径差
    expanded_db = frequency_response(b_eq, a_eq, [w])[0].magnitude_db
    assert abs(expanded_db - ref_db) > abs(db_value - ref_db)

    # 通带里展开 Horner 直接崩掉（真值约 0 dB，展开给出 -200 dB 以下）
    w_pass = 0.05 * math.pi
    z = cmath.exp(1j * w_pass)
    product = 1.0 + 0j
    for segment in segments:
        product *= _transfer_from_coefficients(segment.b, segment.a, z)
    pass_ref_db = 20.0 * math.log10(abs(product))
    assert abs(pass_ref_db) < 0.1
    chain_pass_db, _ = chain_db_phase(segments, w_pass)
    assert chain_pass_db == pytest.approx(pass_ref_db, abs=1e-9)
    expanded_pass_db = frequency_response(b_eq, a_eq, [w_pass])[0].magnitude_db
    assert expanded_pass_db < -100.0

    # 38 阶展开分母直接求根会误判（数值风险的存在性证明）
    expanded_roots = roots_polynomial(a_eq)
    false_outside = [r for r in expanded_roots if abs(r) > 1.0 + 1e-6]
    assert false_outside  # 存在被误算到圆外的稳定极点
    assert max(abs(r) for r in false_outside) > 1.3
    # 而分段报告全部判在圆内
    assert stability_report(segments) == []


# ---------------------------------------------------------------------------
# 关系四：单段退化逐位相等
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        [{"type": "iir", "order": 5, "cutoff": 0.3}],
        [{"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}],
        [{"type": "coefficients", "b": [1.0, -0.6, 0.2], "a": [1.0, 0.3]}],
    ],
)
def test_single_segment_chain_is_bitwise_identical(payload):
    segment = resolve(payload)[0]
    b_eq, a_eq = cascade_coefficients(resolve(payload))
    assert b_eq == segment.b
    assert a_eq == segment.a


def test_single_segment_response_and_stability_identical():
    segment = resolve([{"type": "iir", "order": 4, "cutoff": 0.25}])[0]
    frequencies = linspace(0.0, math.pi, 41)
    points = chain_response_points([segment], frequencies)
    direct = frequency_response(segment.b, segment.a, frequencies)
    for (_w, db_value, phase), ref in zip(points, direct, strict=True):
        assert db_value == ref.magnitude_db
        assert phase == pytest.approx(ref.phase_rad, abs=1e-12)
    assert stability_report([segment]) == []


# ---------------------------------------------------------------------------
# 关系五：阻带衰减方向性
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "extra",
    [
        {"type": "iir", "order": 4, "cutoff": 0.2},
        {"type": "fir", "order": 12, "cutoff": 0.3, "window": "hamming"},
    ],
)
def test_appending_lowpass_only_deepens_stopband_attenuation(extra):
    base_payload = [{"type": "iir", "order": 4, "cutoff": 0.2}]
    base = resolve(base_payload)
    longer = resolve(base_payload + [extra])
    passband, stopband = (0.0, 0.15), (0.45, 1.0)

    before = compute_chain_metrics(
        base, passband, stopband, stability_report(base)
    )
    after = compute_chain_metrics(
        longer, passband, stopband, stability_report(longer)
    )
    assert after["stopband_attenuation_db"] >= before["stopband_attenuation_db"]

    # 逐频点单调：阻带内每个点 |H_total| <= |H_base|（再串的也是低通，
    # 但该段在高频未必处处 <1，因此只断言最坏点指标，不逐点断言）
    assert after["stopband_worst_db"] <= before["stopband_worst_db"] + 1e-12


def test_doubling_identical_iir_doubles_stopband_attenuation():
    """同一段 IIR 串两次：衰减按 dB 叠加（约两倍），不是各段指标的简单 max。"""
    payload = [{"type": "iir", "order": 4, "cutoff": 0.2}]
    once = resolve(payload)
    twice = resolve(payload * 2)
    passband, stopband = (0.0, 0.15), (0.4, 1.0)
    m1 = compute_chain_metrics(once, passband, stopband, stability_report(once))
    m2 = compute_chain_metrics(twice, passband, stopband, stability_report(twice))
    # 巴特沃斯单调：两段的最坏点都在阻带下沿，dB 直接相加
    assert m2["stopband_attenuation_db"] == pytest.approx(
        2.0 * m1["stopband_attenuation_db"], rel=1e-3
    )
    assert m2["stopband_attenuation_db"] > m1["stopband_attenuation_db"]


# ---------------------------------------------------------------------------
# 指标本身的物理合理性
# ---------------------------------------------------------------------------

def test_iir_mixed_chain_phase_reported_nonlinear_with_deviation():
    """混进 IIR 后相位不严格线性：必须如实报 linear_phase=False 和偏离量。"""
    segments = resolve(mixed_chain_payload())
    metrics = compute_chain_metrics(
        segments, (0.0, 0.12), (0.45, 1.0), stability_report(segments)
    )
    assert metrics["linear_phase"] is False
    assert metrics["phase_max_deviation_rad"] > 0.05
    assert metrics["group_delay_variation_samples"] > 0.0
    assert metrics["stable"] is True


def test_unstable_pole_attributed_to_segment():
    segments = resolve([
        {"type": "iir", "order": 2, "cutoff": 0.3},
        # 分母 z^2 - 1.5 z + 0.4 = (z - 1.153)(z - 0.347)：一个圆外极点
        {"type": "coefficients", "b": [1.0], "a": [1.0, -1.5, 0.4]},
        {"type": "fir", "order": 6, "cutoff": 0.3, "window": "hamming"},
    ])
    report = stability_report(segments)
    assert report
    assert {p["segment"] for p in report} == {2}
    metrics = compute_chain_metrics(
        segments, (0.0, 0.1), (0.5, 1.0), report
    )
    assert metrics["stable"] is False
    assert metrics["unstable_segments"] == [2]


# ---------------------------------------------------------------------------
# 关系六：非法输入一律拒绝（中文原因）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "segments",
    [
        None,
        [],
        [{"type": "wavelet", "order": 2, "cutoff": 0.3}],
        [{"order": 2, "cutoff": 0.3}],  # 缺 type
        [{"type": "iir", "order": 0, "cutoff": 0.3}],
        [{"type": "iir", "order": 17, "cutoff": 0.3}],
        [{"type": "iir", "order": 4, "cutoff": 0.0}],
        [{"type": "iir", "order": 4, "cutoff": 1.0}],
        [{"type": "iir", "order": 4, "cutoff": 0.3, "window": "hamming"}],
        [{"type": "fir", "order": 4, "cutoff": 0.3}],  # FIR 缺窗
        [{"type": "fir", "order": 4, "cutoff": 0.3, "window": "blackman"}],
        [{"type": "coefficients", "b": [1.0, "x"], "a": [1.0]}],
        [{"type": "coefficients", "b": [1.0], "a": [1.0, None]}],
        [{"type": "coefficients", "b": [1.0], "a": [0.0, 0.0]}],
        [{"type": "coefficients", "b": [], "a": [1.0]}],
    ],
)
def test_resolve_segments_rejects_invalid(segments):
    with pytest.raises(FilterError) as excinfo:
        resolve(segments)
    assert str(excinfo.value)  # 必须带中文原因


def test_empty_chain_coefficients_rejected():
    with pytest.raises(FilterError):
        cascade_coefficients([])
    with pytest.raises(FilterError):
        chain_response_points([], [0.1])


@pytest.mark.parametrize(
    "band",
    [
        None,
        [0.1],
        [0.1, 0.2, 0.3],
        [0.5, 0.2],   # 给反
        [0.3, 0.3],   # 相等
        [-0.1, 0.5],
        [0.5, 1.2],
        [0.5, "x"],
        [True, 0.5],
    ],
)
def test_validate_band_rejects_invalid(band):
    with pytest.raises(FilterError):
        validate_band(band, "测试频带")


def test_validate_band_accepts_boundaries():
    assert validate_band([0.0, 0.5], "通带") == (0.0, 0.5)
    assert validate_band([0.5, 1.0], "阻带") == (0.5, 1.0)
