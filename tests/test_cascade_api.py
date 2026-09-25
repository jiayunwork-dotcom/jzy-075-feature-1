"""级联链路 HTTP 接口测试：POST /api/v1/filters/cascade。

覆盖接口形状、等效系数可喂回既有频响 / 零极点接口、链路指标结构、
稳定性归属，以及非法输入的 400 + 中文原因。
核心数学关系在 tests/test_cascade.py 逐关系验证，本文件只打接口层。
"""

from __future__ import annotations

import math

import pytest

PASSBAND = [0.0, 0.15]
STOPBAND = [0.45, 1.0]


def mixed_payload():
    return {
        "segments": [
            {"type": "iir", "order": 4, "cutoff": 0.2},
            {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"},
            {"type": "coefficients", "b": [1.0, -0.5], "a": [1.0]},
        ],
        "passband": PASSBAND,
        "stopband": STOPBAND,
    }


def test_cascade_returns_equivalent_coefficients_and_segments(client):
    response = client.post("/api/v1/filters/cascade", json=mixed_payload())
    assert response.status_code == 200, response.text
    data = response.json()

    # 等效系数格式与单滤波器设计结果完全一致
    assert set(data.keys()) == {"b", "a", "segments", "metrics"}
    assert all(isinstance(v, float) for v in data["b"])
    assert all(isinstance(v, float) for v in data["a"])
    # IIR4(5 抽头) * FIR12(13 抽头) * 一阶(2 抽头) => 18 个分子系数
    assert len(data["b"]) == 5 + 13 + 2 - 2
    # 只有 IIR 段有 4 阶分母，其余两段分母为常数
    assert len(data["a"]) == 5

    # 段回显：三段、按请求次序、设计意图被现场落地为系数
    segments = data["segments"]
    assert [s["position"] for s in segments] == [1, 2, 3]
    assert [s["type"] for s in segments] == ["iir", "fir", "coefficients"]
    assert segments[0]["spec"] == {"order": 4, "cutoff": 0.2}
    assert segments[1]["spec"] == {
        "order": 12,
        "cutoff": 0.35,
        "window": "hamming",
    }
    assert len(segments[1]["b"]) == 13
    assert segments[1]["a"] == [1.0]


def test_cascade_equivalent_coefficients_feed_back_to_frequency(client):
    data = client.post("/api/v1/filters/cascade", json=mixed_payload()).json()
    frequencies = [0.0, 0.1 * math.pi, 0.2 * math.pi]
    response = client.post(
        "/api/v1/filters/frequency",
        json={"b": data["b"], "a": data["a"], "frequencies": frequencies},
    )
    assert response.status_code == 200, response.text
    points = response.json()["points"]
    assert len(points) == 3
    # 三条低通链在直流处增益接近 1（FIR 略偏、IIR 归一化、一阶系数段为 0.5）
    # 系数段 b=[1,-0.5] 直流增益 0.5，故整体直流约 -6 dB
    assert points[0]["magnitude_db"] == pytest.approx(-6.0206, abs=0.1)


def test_cascade_equivalent_coefficients_feed_back_to_zero_poles(client):
    data = client.post("/api/v1/filters/cascade", json=mixed_payload()).json()
    response = client.post(
        "/api/v1/filters/zero-poles",
        json={"b": data["b"], "a": data["a"]},
    )
    assert response.status_code == 200, response.text
    zp = response.json()
    assert len(zp["poles"]) == 4
    assert zp["stable"] is True


def test_cascade_metrics_structure_and_sanity(client):
    data = client.post("/api/v1/filters/cascade", json=mixed_payload()).json()
    metrics = data["metrics"]
    expected_keys = {
        "reference_level_db",
        "reference_frequency",
        "passband_ripple_db",
        "passband_max_db",
        "passband_min_db",
        "passband_min_frequency",
        "stopband_attenuation_db",
        "stopband_worst_db",
        "stopband_worst_frequency",
        "group_delay_samples",
        "group_delay_variation_samples",
        "phase_max_deviation_rad",
        "phase_rms_deviation_rad",
        "linear_phase",
        "stable",
        "unstable_segments",
        "unstable_poles",
    }
    assert set(metrics.keys()) == expected_keys
    assert metrics["passband_ripple_db"] >= 0.0
    assert metrics["stopband_attenuation_db"] > 0.0
    assert 0.0 <= metrics["reference_frequency"] <= 0.15
    assert 0.45 <= metrics["stopband_worst_frequency"] <= 1.0
    assert metrics["stable"] is True
    assert metrics["unstable_segments"] == []
    assert metrics["unstable_poles"] == []
    # 混了 IIR 段，如实报非线性并给出偏离量
    assert metrics["linear_phase"] is False
    assert metrics["phase_max_deviation_rad"] > 0.0
    assert metrics["group_delay_samples"] > 0.0


def test_cascade_pure_fir_chain_reports_linear_phase(client):
    payload = {
        "segments": [
            {"type": "fir", "order": 12, "cutoff": 0.3, "window": "hamming"},
            {"type": "fir", "order": 8, "cutoff": 0.35, "window": "hann"},
        ],
        "passband": [0.0, 0.2],
        "stopband": [0.5, 1.0],
    }
    metrics = client.post("/api/v1/filters/cascade", json=payload).json()["metrics"]
    assert metrics["linear_phase"] is True
    # 群延时 = (12 + 8) / 2 = 10 个采样
    assert metrics["group_delay_samples"] == pytest.approx(10.0, abs=1e-6)


def test_cascade_high_order_stable_chain_via_http(client):
    """三段 IIR 共 38 阶：不能因阶数高误判不稳定。"""
    payload = {
        "segments": [
            {"type": "iir", "order": 16, "cutoff": 0.1},
            {"type": "iir", "order": 16, "cutoff": 0.12},
            {"type": "iir", "order": 6, "cutoff": 0.2},
        ],
        "passband": [0.0, 0.08],
        "stopband": [0.3, 1.0],
    }
    response = client.post("/api/v1/filters/cascade", json=payload)
    assert response.status_code == 200, response.text
    metrics = response.json()["metrics"]
    assert metrics["stable"] is True
    assert metrics["unstable_segments"] == []
    # 深阻带链路压制很强，指标必须是有限实数而不是 NaN/Inf
    assert math.isfinite(metrics["stopband_attenuation_db"])
    assert metrics["stopband_attenuation_db"] > 200.0


def test_cascade_reports_which_segment_is_unstable(client):
    payload = {
        "segments": [
            {"type": "iir", "order": 2, "cutoff": 0.3},
            {"type": "coefficients", "b": [1.0], "a": [1.0, -1.5, 0.4]},
            {"type": "fir", "order": 6, "cutoff": 0.3, "window": "hamming"},
        ],
        "passband": [0.0, 0.1],
        "stopband": [0.5, 1.0],
    }
    response = client.post("/api/v1/filters/cascade", json=payload)
    assert response.status_code == 200, response.text
    metrics = response.json()["metrics"]
    assert metrics["stable"] is False
    assert metrics["unstable_segments"] == [2]
    pole = metrics["unstable_poles"][0]
    assert pole["segment"] == 2
    assert pole["radius"] == pytest.approx(1.153, abs=1e-3)
    assert pole["distance_to_unit_circle"] > 0.1


def test_cascade_single_segment_bitwise_equal_to_design(client):
    designed = client.post(
        "/api/v1/filters/design",
        json={"type": "iir", "order": 4, "cutoff": 0.25},
    ).json()
    cascaded = client.post(
        "/api/v1/filters/cascade",
        json={
            "segments": [{"type": "iir", "order": 4, "cutoff": 0.25}],
            "passband": [0.0, 0.15],
            "stopband": [0.4, 1.0],
        },
    ).json()
    assert cascaded["b"] == designed["b"]
    assert cascaded["a"] == designed["a"]


# ---------------------------------------------------------------------------
# 非法输入：400 + 中文原因
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "segments",
    [
        [],
        [{"type": "wavelet", "order": 2, "cutoff": 0.3}],
        [{"type": "iir", "order": 17, "cutoff": 0.3}],
        [{"type": "iir", "order": 4, "cutoff": 1.0}],
        [{"type": "iir", "order": 4, "cutoff": 0.3, "window": "hamming"}],
        [{"type": "fir", "order": 4, "cutoff": 0.3}],
        [{"type": "coefficients", "b": [1.0, "x"], "a": [1.0]}],
        [{"type": "coefficients", "b": [1.0], "a": [0.0, 0.0]}],
    ],
)
def test_cascade_rejects_bad_segments(client, segments):
    payload = {"segments": segments, "passband": PASSBAND, "stopband": STOPBAND}
    response = client.post("/api/v1/filters/cascade", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]


@pytest.mark.parametrize(
    "passband,stopband",
    [
        ([0.2, 0.05], [0.5, 1.0]),      # 通带给反
        ([0.0, 0.15], [0.9, 0.5]),      # 阻带给反
        ([0.0, 0.15], [-0.1, 0.5]),     # 阻带越下界
        ([0.0, 0.15], [0.5, 1.2]),      # 阻带越上界
        ([-0.1, 0.5], [0.6, 1.0]),      # 通带越界
        ([0.3, 0.3], [0.5, 1.0]),       # 通带两端相等
        ([0.0, 0.15], [0.5]),           # 阻带缺元素
        ([0.0, "x"], [0.5, 1.0]),       # 掺非数值
    ],
)
def test_cascade_rejects_bad_bands(client, passband, stopband):
    payload = {
        "segments": [{"type": "iir", "order": 2, "cutoff": 0.3}],
        "passband": passband,
        "stopband": stopband,
    }
    response = client.post("/api/v1/filters/cascade", json=payload)
    assert response.status_code == 400, response.text
    assert response.json()["error"]


def test_cascade_missing_top_level_fields_is_422(client):
    response = client.post("/api/v1/filters/cascade", json={"segments": []})
    assert response.status_code == 422


def test_cascade_segments_not_an_array_is_422(client):
    response = client.post(
        "/api/v1/filters/cascade",
        json={"segments": {"type": "iir"}, "passband": PASSBAND, "stopband": STOPBAND},
    )
    assert response.status_code == 422
