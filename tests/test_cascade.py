"""级联链路合成与分析的自动化测试（HTTP 层）。

盯住需求里点名的六条关系：
1. 等效系数正确性：等效 b/a 进频响接口 == 各段频响逐点复数相乘；
2. 级联顺序无关性：同样几段换次序，等效频响完全一致；
3. 数值稳定性：二十阶以上的高阶链路，圆内极点不得被误判到圆外；
4. 退化情形：单段链路的等效系数与该段逐位相等；
5. 阻带衰减方向性：再串一段同型低通，阻带最坏衰减只深不浅；
6. 非法输入一律 400 + 中文原因。

另附链路级指标行为：纯 FIR 链相位线性且群延时为各段之和、混进 IIR
后如实报告非线性、不稳定段被点名归因、等效系数能喂回已有接口复核。
"""

from __future__ import annotations

import cmath
import math

import pytest

from app.butterworth import (
    bilinear_map,
    butterworth_prototype_poles,
    prewarp,
)
from tests.conftest import fir_coeffs, iir_coeffs, linspace

_VALID_SEGMENT = {"design": {"type": "iir", "order": 2, "cutoff": 0.3}}
_VALID_BANDS = {"passband": [0.0, 0.1], "stopband": [0.4, 1.0]}


def _analyze(client, payload):
    response = client.post("/api/v1/cascade/analyze", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _frequency_points(client, b, a, freqs):
    response = client.post(
        "/api/v1/filters/frequency",
        json={"b": b, "a": a, "frequencies": freqs},
    )
    assert response.status_code == 200, response.text
    return response.json()["points"]


def _point_to_complex(point):
    amplitude = 10.0 ** (point["magnitude_db"] / 20.0)
    return amplitude * cmath.exp(1j * point["phase_rad"])


# ---------------------------------------------------------------- 关系一


def test_equivalent_coefficients_reproduce_stage_product(client):
    """等效系数的频响 == 每段单独求频响再逐点复数相乘（浮点容差内）。"""
    iir_b, iir_a = iir_coeffs(client, 4, 0.45)
    payload = {
        "segments": [
            {"design": {"type": "iir", "order": 4, "cutoff": 0.2}},
            {"design": {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}},
            {"coefficients": {"b": iir_b, "a": iir_a}},
        ]
    }
    data = _analyze(client, payload)
    equivalent = data["equivalent"]
    # 等效阶数 = 各段阶数之和，证明确实走了卷积合成
    assert equivalent["numerator_order"] == 4 + 12 + 4
    assert equivalent["denominator_order"] == 4 + 4

    freqs = linspace(0.0, math.pi, 257)
    eq_points = _frequency_points(client, equivalent["b"], equivalent["a"], freqs)
    segment_points = [
        _frequency_points(client, segment["b"], segment["a"], freqs)
        for segment in data["segments"]
    ]
    for i, w in enumerate(freqs):
        product = 1.0 + 0.0j
        for points in segment_points:
            product *= _point_to_complex(points[i])
        product_db = 20.0 * math.log10(abs(product)) if abs(product) > 0 else -1000.0
        if product_db > -40.0:
            # 通带 / 过渡带：复数响应逐点一致
            eq_value = _point_to_complex(eq_points[i])
            assert abs(eq_value - product) <= 1e-9 * abs(product), f"w={w}"
        if product_db > -120.0:
            # 更深的区域只比幅度 dB（深阻带的相位无意义）
            assert eq_points[i]["magnitude_db"] == pytest.approx(product_db, abs=1e-4)


# ---------------------------------------------------------------- 关系二


def test_cascade_order_does_not_change_equivalent_response(client):
    """同样几段换个先后次序：等效系数（容差内）与等效频响完全一致。"""
    iir_b, iir_a = iir_coeffs(client, 3, 0.45)
    seg_iir = {"design": {"type": "iir", "order": 4, "cutoff": 0.2}}
    seg_fir = {"design": {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}}
    seg_coeffs = {"coefficients": {"b": iir_b, "a": iir_a}}
    forward = _analyze(
        client, {"segments": [seg_iir, seg_fir, seg_coeffs], **_VALID_BANDS}
    )
    backward = _analyze(
        client, {"segments": [seg_coeffs, seg_fir, seg_iir], **_VALID_BANDS}
    )

    # 卷积顺序不同只引入末位浮点差异
    for x, y in zip(forward["equivalent"]["b"], backward["equivalent"]["b"], strict=True):
        assert x == pytest.approx(y, rel=1e-12, abs=1e-18)
    for x, y in zip(forward["equivalent"]["a"], backward["equivalent"]["a"], strict=True):
        assert x == pytest.approx(y, rel=1e-12, abs=1e-18)

    # 等效频响完全一致
    freqs = linspace(0.0, math.pi, 200)
    fwd_db = [
        p["magnitude_db"]
        for p in _frequency_points(client, forward["equivalent"]["b"], forward["equivalent"]["a"], freqs)
    ]
    bwd_db = [
        p["magnitude_db"]
        for p in _frequency_points(client, backward["equivalent"]["b"], backward["equivalent"]["a"], freqs)
    ]
    for x, y in zip(fwd_db, bwd_db, strict=True):
        if max(x, y) > -120.0:
            assert x == pytest.approx(y, abs=1e-9)

    # 链路级指标同样与次序无关
    fwd_metrics, bwd_metrics = forward["metrics"], backward["metrics"]
    assert fwd_metrics["passband"]["max_deviation_db"] == pytest.approx(
        bwd_metrics["passband"]["max_deviation_db"], abs=1e-9
    )
    assert fwd_metrics["stopband"]["attenuation_db"] == pytest.approx(
        bwd_metrics["stopband"]["attenuation_db"], abs=1e-9
    )
    assert fwd_metrics["phase"]["group_delay_samples"] == pytest.approx(
        bwd_metrics["phase"]["group_delay_samples"], abs=1e-9
    )


# ---------------------------------------------------------------- 关系三


def test_high_order_chain_poles_stay_inside_unit_circle(client):
    """四段 6 阶 IIR（截止贴近直流）串成 24 阶链路：极点必须全部判在圆内。"""
    cutoffs = (0.05, 0.08, 0.1, 0.15)
    payload = {
        "segments": [
            {"design": {"type": "iir", "order": 6, "cutoff": cutoff}}
            for cutoff in cutoffs
        ],
        "passband": [0.0, 0.03],
        "stopband": [0.2, 1.0],
    }
    data = _analyze(client, payload)
    assert data["equivalent"]["denominator_order"] == 24  # 确实是高阶链路

    stability = data["metrics"]["stability"]
    assert stability["stable"] is True
    assert stability["introduced_by_segments"] == []
    poles = stability["poles"]
    assert len(poles) == 24
    for pole in poles:
        assert pole["radius"] < 1.0
        assert pole["distance_to_unit_circle"] < 0.0
        assert pole["on_unit_circle"] is False

    # 与解析极点（巴特沃斯原型 -> 预畸变 -> 双线性）逐根配对
    analytic = []
    for cutoff in cutoffs:
        omega = prewarp(cutoff)
        analytic.extend(
            bilinear_map(p * omega) for p in butterworth_prototype_poles(6)
        )
    remaining = list(analytic)
    for pole in poles:
        value = complex(pole["real"], pole["imag"])
        nearest = min(remaining, key=lambda target: abs(value - target))
        assert abs(value - nearest) < 1e-7
        remaining.remove(nearest)


# ---------------------------------------------------------------- 关系四


def test_single_segment_chain_is_bit_identical(client):
    """单段链路：等效系数与该段本身逐位相等，不因合成流程引入偏差。"""
    b, a = iir_coeffs(client, 5, 0.3)
    data = _analyze(client, {"segments": [{"coefficients": {"b": b, "a": a}}]})
    assert data["equivalent"]["b"] == b  # 逐位相等，不是近似
    assert data["equivalent"]["a"] == a

    # 设计意图单段：等效系数与设计接口直接给出的完全一致
    data2 = _analyze(
        client,
        {"segments": [{"design": {"type": "fir", "order": 6, "cutoff": 0.4, "window": "hann"}}]},
    )
    expected_b, expected_a = fir_coeffs(client, 6, 0.4, window="hann")
    assert data2["equivalent"]["b"] == expected_b
    assert data2["equivalent"]["a"] == expected_a


# ---------------------------------------------------------------- 关系五


def test_extra_lowpass_stage_never_shallows_stopband(client):
    """往链路里再串一段同型低通：指定阻带的最坏衰减只能更深或持平。"""
    base_segments = [
        {"design": {"type": "iir", "order": 4, "cutoff": 0.2}},
        {"design": {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}},
    ]
    bands = {"passband": [0.0, 0.1], "stopband": [0.4, 1.0]}
    base = _analyze(client, {"segments": base_segments, **bands})
    extended = _analyze(
        client,
        {
            "segments": base_segments
            + [{"design": {"type": "iir", "order": 4, "cutoff": 0.15}}],
            **bands,
        },
    )
    attenuation_base = base["metrics"]["stopband"]["attenuation_db"]
    attenuation_extended = extended["metrics"]["stopband"]["attenuation_db"]
    assert attenuation_extended >= attenuation_base - 1e-9
    # 同型低通串联，阻带应当明显加深，而不是原地踏步
    assert attenuation_extended > attenuation_base + 10.0


# ---------------------------------------------------------------- 关系六


@pytest.mark.parametrize(
    "payload",
    [
        {},  # 缺 segments
        {"segments": []},  # 空链路
        {"segments": "not-a-list"},
        {"segments": [{}]},  # 段里既没有 design 也没有 coefficients
        {
            "segments": [
                {
                    "design": {"type": "iir", "order": 2, "cutoff": 0.3},
                    "coefficients": {"b": [1.0], "a": [1.0]},
                }
            ]
        },  # design 与 coefficients 同时给
        {"segments": [{"coefficients": {"b": [1.0, "x"], "a": [1.0]}}]},  # 非数值
        {"segments": [{"coefficients": {"b": [1.0], "a": [0.0, 0.0]}}]},  # 分母全零
        {"segments": [{"coefficients": {"b": [0.0, 0.0], "a": [1.0]}}]},  # 分子全零
        {"segments": [{"coefficients": {"b": [1.0]}}]},  # 缺 a
        {"segments": [{"design": {"type": "iir", "order": 0, "cutoff": 0.3}}]},
        {"segments": [{"design": {"type": "iir", "order": 17, "cutoff": 0.3}}]},
        {"segments": [{"design": {"type": "iir", "order": 4, "cutoff": -0.5}}]},
        {
            "segments": [
                {"design": {"type": "fir", "order": 4, "cutoff": 1.0, "window": "hamming"}}
            ]
        },
        {"segments": [{"design": {"type": "fir", "order": 4, "cutoff": 0.3}}]},  # 缺窗
        {
            "segments": [
                {"design": {"type": "iir", "order": 4, "cutoff": 0.3, "window": "hann"}}
            ]
        },  # IIR 不接受窗
        {"segments": [{"design": {"type": "wavelet", "order": 4, "cutoff": 0.3}}]},
        {
            "segments": [
                {"design": {"type": "fir", "order": 4, "cutoff": 0.3, "window": "blackman"}}
            ]
        },
        {"segments": [_VALID_SEGMENT], "passband": [0.5, 0.2], "stopband": [0.6, 1.0]},  # 给反
        {"segments": [_VALID_SEGMENT], "passband": [-0.1, 0.2], "stopband": [0.6, 1.0]},  # 越界
        {"segments": [_VALID_SEGMENT], "passband": [0.0, 0.2], "stopband": [0.6, 1.5]},  # 越界
        {"segments": [_VALID_SEGMENT], "passband": [0.0, 0.2]},  # 只给一个带
        {"segments": [_VALID_SEGMENT], **_VALID_BANDS, "grid_points": 10},  # 太少
        {"segments": [_VALID_SEGMENT], **_VALID_BANDS, "grid_points": 100.5},  # 非整数
    ],
)
def test_cascade_rejects_invalid_inputs(client, payload):
    response = client.post("/api/v1/cascade/analyze", json=payload)
    assert response.status_code == 400, response.text
    assert response.json()["error"]  # 必须带中文原因


def test_empty_chain_error_message_is_chinese(client):
    response = client.post("/api/v1/cascade/analyze", json={"segments": []})
    assert response.status_code == 400
    assert "空" in response.json()["error"]


def test_reversed_band_error_message_is_chinese(client):
    response = client.post(
        "/api/v1/cascade/analyze",
        json={"segments": [_VALID_SEGMENT], "passband": [0.5, 0.2], "stopband": [0.6, 1.0]},
    )
    assert response.status_code == 400
    assert "通带" in response.json()["error"]


# ------------------------------------------------------- 链路级指标行为


def test_fir_only_chain_has_linear_phase_and_summed_group_delay(client):
    """纯 FIR 链路：相位仍接近线性，群延时 = 各段群延时之和。"""
    payload = {
        "segments": [
            {"design": {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}},
            {"design": {"type": "fir", "order": 10, "cutoff": 0.4, "window": "hamming"}},
            {"design": {"type": "fir", "order": 8, "cutoff": 0.45, "window": "hann"}},
        ],
        "passband": [0.0, 0.2],
        "stopband": [0.6, 1.0],
    }
    phase = _analyze(client, payload)["metrics"]["phase"]
    assert phase["linear"] is True
    assert phase["contains_recursive_stage"] is False
    # 群延时 = (12 + 10 + 8) / 2 = 15 个采样
    assert phase["group_delay_samples"] == pytest.approx(15.0, abs=1e-6)


def test_mixed_chain_phase_nonlinearity_reported_honestly(client):
    """链路里混进 IIR：相位不再严格线性，必须如实报出偏离程度。"""
    payload = {
        "segments": [
            {"design": {"type": "iir", "order": 4, "cutoff": 0.2}},
            {"design": {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}},
        ],
        "passband": [0.0, 0.15],
        "stopband": [0.45, 1.0],
    }
    metrics = _analyze(client, payload)["metrics"]
    phase = metrics["phase"]
    assert phase["contains_recursive_stage"] is True
    assert phase["linear"] is False
    assert phase["max_deviation_rad"] > 0.05
    assert 10.0 < phase["group_delay_samples"] < 16.0
    # 指标基于合成后的等效系统，直接给出合理量级
    assert metrics["passband"]["max_deviation_db"] >= 0.0
    assert metrics["stopband"]["attenuation_db"] > 30.0


def test_unstable_segment_is_identified_by_index(client):
    """任何一段贡献单位圆外极点：整条链路判不稳定，并点名是哪一段。"""
    stable_b, stable_a = iir_coeffs(client, 3, 0.25)
    payload = {
        "segments": [
            {"coefficients": {"b": stable_b, "a": stable_a}},
            {"coefficients": {"b": [1.0], "a": [1.0, -1.3]}},  # 极点 1.3 在圆外
            {"design": {"type": "fir", "order": 4, "cutoff": 0.5, "window": "hamming"}},
        ],
        **_VALID_BANDS,
    }
    stability = _analyze(client, payload)["metrics"]["stability"]
    assert stability["stable"] is False
    assert stability["introduced_by_segments"] == [2]
    assert [s["stable"] for s in stability["segments"]] == [True, False, True]
    bad_poles = [p for p in stability["poles"] if p["segment_index"] == 2]
    assert len(bad_poles) == 1
    assert bad_poles[0]["radius"] == pytest.approx(1.3)


def test_response_records_segment_order_for_audit(client):
    """结果里要能反映链路由哪几段、按什么次序拼起来，方便对账。"""
    payload = {
        "segments": [
            {"name": "压工频", "design": {"type": "iir", "order": 4, "cutoff": 0.2}},
            {"name": "削毛刺", "design": {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}},
            {"coefficients": {"b": [1.0, -0.5], "a": [1.0]}},
        ]
    }
    data = _analyze(client, payload)
    assert data["segment_count"] == 3
    segments = data["segments"]
    assert [s["index"] for s in segments] == [1, 2, 3]
    assert [s["name"] for s in segments] == ["压工频", "削毛刺", "段3"]
    assert [s["source"] for s in segments] == ["design", "design", "coefficients"]
    assert segments[0]["design"]["type"] == "iir"
    assert segments[1]["design"]["window"] == "hamming"
    # 设计意图段落定的系数与设计接口一致，可直接对账
    designed = client.post(
        "/api/v1/filters/design",
        json={"type": "iir", "order": 4, "cutoff": 0.2},
    ).json()
    assert segments[0]["b"] == designed["b"]
    assert segments[0]["a"] == designed["a"]


def test_equivalent_coefficients_feed_back_into_existing_endpoints(client):
    """等效系数格式与单滤波器一致，可直接喂回频响 / 零极点接口复核。"""
    data = _analyze(
        client,
        {
            "segments": [
                {"design": {"type": "iir", "order": 2, "cutoff": 0.25}},
                {"design": {"type": "iir", "order": 2, "cutoff": 0.4}},
            ]
        },
    )
    equivalent = data["equivalent"]
    zp = client.post(
        "/api/v1/filters/zero-poles",
        json={"b": equivalent["b"], "a": equivalent["a"]},
    )
    assert zp.status_code == 200, zp.text
    assert zp.json()["stable"] is True
    assert len(zp.json()["poles"]) == 4

    freq = client.post(
        "/api/v1/filters/frequency",
        json={"b": equivalent["b"], "a": equivalent["a"], "frequencies": [0.0]},
    )
    assert freq.status_code == 200
    # 两级巴特沃斯直流增益都是 1，合成后直流仍应为 0 dB
    assert freq.json()["points"][0]["magnitude_db"] == pytest.approx(0.0, abs=1e-9)
