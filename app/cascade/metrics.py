"""链路级指标：通带纹波、阻带衰减、相位线性度 / 群延时、稳定性。

所有指标都从**合成后的等效系统**出发计算，绝不把各段指标简单相加
（纹波与衰减在级联时会叠加、甚至相互抵消）。实现上：

- 幅频 / 相频采样走 chain_transfer 的逐段求值路径——它与等效传递
  函数数学上等价，但避开了高阶展开多项式的相消误差；
- 极点逐段求解再按段归因（高阶合成多项式求根会把圆内极点推到圆外）。

频带以相对奈奎斯特的归一化值给出，0 和 1 分别对应直流与奈奎斯特
（数字角频率 0 与 pi rad/采样）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..phase import principal_phase, unwrap_phase
from .chain import Stage, chain_transfer
from .sections import stage_poles

_MIN_DB = -1000.0  # 传输值数值为 0 时的兜底下限
# 极点稳定性容差，与单滤波器零极点分析保持一致
_ON_CIRCLE_TOL = 1e-9
# 通带相位对最佳线性拟合的最大残差低于此值判为"接近线性"（弧度）
PHASE_LINEARITY_TOL = 0.02


def _to_db(amplitude: float) -> float:
    if amplitude <= 1e-300:
        return _MIN_DB
    return 20.0 * math.log10(amplitude)


def _band_grid(low: float, high: float, grid_points: int) -> list[float]:
    """频带内的等间隔数字角频率采样点（含端点）。

    点数按带宽占奈奎斯特的比例从总采样点数分配，最少 8 点；
    零宽度频带向两侧各扩 1e-6（裁剪到 [0, 1]），保证可采样。
    """
    if high == low:
        low = max(0.0, low - 1e-6)
        high = min(1.0, high + 1e-6)
    count = max(8, min(grid_points, int(math.ceil(grid_points * (high - low)))))
    step = (high - low) / (count - 1)
    return [(low + step * k) * math.pi for k in range(count)]


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """最小二乘直线拟合，返回 (斜率, 截距)。"""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    return slope, mean_y - slope * mean_x


@dataclass(frozen=True)
class BandSampling:
    frequencies: list[float]
    magnitude_db: list[float]
    phase_rad: list[float]  # 已展开


def _sample(stages: list[Stage], frequencies: list[float]) -> BandSampling:
    magnitudes: list[float] = []
    raw_phases: list[float] = []
    for frequency in frequencies:
        value = chain_transfer(stages, frequency)
        magnitudes.append(_to_db(abs(value)))
        raw_phases.append(principal_phase(value))
    return BandSampling(
        frequencies=frequencies,
        magnitude_db=magnitudes,
        phase_rad=unwrap_phase(raw_phases),
    )


def _describe_pole(pole: complex, segment_index: int) -> dict[str, object]:
    radius = abs(pole)
    return {
        "segment_index": segment_index,
        "real": pole.real,
        "imag": pole.imag,
        "radius": radius,
        "distance_to_unit_circle": radius - 1.0,
        "on_unit_circle": abs(radius - 1.0) < _ON_CIRCLE_TOL,
    }


def compute_metrics(
    stages: list[Stage],
    passband: tuple[float, float],
    stopband: tuple[float, float],
    grid_points: int,
) -> dict[str, object]:
    """计算链路整体的纹波、衰减、相位 / 群延时与稳定性指标。"""
    # 参考电平取直流（w=0）增益；多级低通链路直流本就无衰减
    reference_db = _to_db(abs(chain_transfer(stages, 0.0)))

    passband_sample = _sample(stages, _band_grid(passband[0], passband[1], grid_points))
    stopband_sample = _sample(stages, _band_grid(stopband[0], stopband[1], grid_points))

    pb = passband_sample.magnitude_db
    passband_metrics = {
        "reference_db": reference_db,
        "max_gain_db": max(pb),
        "min_gain_db": min(pb),
        # 通带增益相对参考电平的最大起伏（分贝）
        "max_deviation_db": max(abs(value - reference_db) for value in pb),
        # 峰峰值：通带内最高到最低的总摆幅
        "peak_to_peak_db": max(pb) - min(pb),
    }

    # 阻带最不衰减的点 = 增益最高点；它决定整条链路顶不顶用
    worst_stopband_db = max(stopband_sample.magnitude_db)
    stopband_metrics = {
        "worst_gain_db": worst_stopband_db,
        "attenuation_db": reference_db - worst_stopband_db,
    }

    # 相位线性度：通带展开相位对角频率做最小二乘直线拟合
    slope, intercept = _fit_line(
        passband_sample.frequencies, passband_sample.phase_rad
    )
    residuals = [
        phase - (intercept + slope * frequency)
        for frequency, phase in zip(
            passband_sample.frequencies, passband_sample.phase_rad, strict=True
        )
    ]
    max_phase_deviation = max(abs(value) for value in residuals)
    rms_phase_deviation = math.sqrt(
        sum(value * value for value in residuals) / len(residuals)
    )
    phase_metrics = {
        "linear": max_phase_deviation <= PHASE_LINEARITY_TOL,
        "max_deviation_rad": max_phase_deviation,
        "rms_deviation_rad": rms_phase_deviation,
        # 群延时 tau = -dphi/dw，单位采样
        "group_delay_samples": -slope,
        "contains_recursive_stage": any(stage.recursive for stage in stages),
    }

    # 稳定性：逐段求根归因。任何一段贡献单位圆外极点，整条链路即不稳定。
    poles: list[dict[str, object]] = []
    segment_stability: list[dict[str, object]] = []
    unstable_segments: list[int] = []
    for stage in stages:
        stage_poles_ = stage_poles(stage.a)
        outside = [
            pole
            for pole in stage_poles_
            if abs(pole) - 1.0 > _ON_CIRCLE_TOL
        ]
        segment_stability.append(
            {
                "index": stage.index,
                "name": stage.name,
                "stable": not outside,
                "pole_count": len(stage_poles_),
                "poles_outside_unit_circle": [
                    _describe_pole(pole, stage.index) for pole in outside
                ],
            }
        )
        if outside:
            unstable_segments.append(stage.index)
        poles.extend(_describe_pole(pole, stage.index) for pole in stage_poles_)

    stability_metrics = {
        "stable": not unstable_segments,
        "introduced_by_segments": unstable_segments,
        "segments": segment_stability,
        "poles": poles,
    }

    return {
        "bands": {"passband": list(passband), "stopband": list(stopband)},
        "grid_points": grid_points,
        "passband": passband_metrics,
        "stopband": stopband_metrics,
        "phase": phase_metrics,
        "stability": stability_metrics,
    }
