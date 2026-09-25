"""链路合成与分析的编排：校验请求 -> 解析段 -> 合成 -> 指标 -> 响应。"""

from __future__ import annotations

import math

from ..errors import FilterError
from .chain import Stage, equivalent_coefficients, resolve_stage
from .metrics import compute_metrics
from .schemas import CascadeRequest

MAX_SEGMENTS = 16  # 链路段数上限，防止失控输入
DEFAULT_GRID_POINTS = 4096
MIN_GRID_POINTS = 32
MAX_GRID_POINTS = 8192


def _validate_segments(raw: object) -> list:
    if raw is None:
        raise FilterError("缺少 segments 字段：链路至少要包含一段")
    if not isinstance(raw, list):
        raise FilterError("segments 必须是段描述数组")
    if len(raw) == 0:
        raise FilterError("链路不能为空：segments 至少要包含一段")
    if len(raw) > MAX_SEGMENTS:
        raise FilterError(f"链路段数 {len(raw)} 超过上限 {MAX_SEGMENTS}")
    return raw


def _validate_band(raw: object, label: str) -> tuple[float, float]:
    """频带：[下限, 上限]，相对奈奎斯特的归一化值，范围 [0, 1]。"""
    if not isinstance(raw, (list, tuple)):
        raise FilterError(f"{label}必须是 [下限, 上限] 两个数值的数组")
    if len(raw) != 2:
        raise FilterError(f"{label}必须恰好包含下限、上限两个数值")
    cleaned: list[float] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise FilterError(f"{label}[{i}] 不是数值")
        value = float(item)
        if not math.isfinite(value):
            raise FilterError(f"{label}[{i}] 必须是有限数值")
        if not (0.0 <= value <= 1.0):
            raise FilterError(
                f"{label}[{i}]={value} 越界：归一化频率（相对奈奎斯特）必须在 [0, 1] 内"
            )
        cleaned.append(value)
    low, high = cleaned
    if low > high:
        raise FilterError(f"{label}范围给反了：下限 {low} 大于上限 {high}")
    return low, high


def _validate_grid_points(raw: object) -> int:
    if isinstance(raw, bool):
        raise FilterError("grid_points 必须是整数")
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    if not isinstance(raw, int):
        raise FilterError("grid_points 必须是整数")
    if not (MIN_GRID_POINTS <= raw <= MAX_GRID_POINTS):
        raise FilterError(
            f"grid_points 必须在 {MIN_GRID_POINTS}..{MAX_GRID_POINTS} 之间，收到 {raw}"
        )
    return raw


def _describe_stage(stage: Stage) -> dict[str, object]:
    """段的对账信息：来源、设计参数（若有）、落定后的系数。"""
    return {
        "index": stage.index,
        "name": stage.name,
        "source": stage.source,
        "design": stage.design,
        "b": stage.b,
        "a": stage.a,
    }


def analyze_cascade(request: CascadeRequest) -> dict[str, object]:
    """把一串滤波段合成等效系统，并计算链路级指标。"""
    segments_raw = _validate_segments(request.segments)

    passband_raw, stopband_raw = request.passband, request.stopband
    if (passband_raw is None) != (stopband_raw is None):
        raise FilterError("passband 与 stopband 必须同时提供，或都不提供")
    passband = stopband = None
    if passband_raw is not None:
        passband = _validate_band(passband_raw, "通带")
        stopband = _validate_band(stopband_raw, "阻带")

    grid_points = (
        DEFAULT_GRID_POINTS
        if request.grid_points is None
        else _validate_grid_points(request.grid_points)
    )

    stages = [resolve_stage(i + 1, raw) for i, raw in enumerate(segments_raw)]
    b_eq, a_eq = equivalent_coefficients(stages)

    result: dict[str, object] = {
        "equivalent": {
            "b": b_eq,
            "a": a_eq,
            "order": max(len(b_eq), len(a_eq)) - 1,
            "numerator_order": len(b_eq) - 1,
            "denominator_order": len(a_eq) - 1,
        },
        "segment_count": len(stages),
        "segments": [_describe_stage(stage) for stage in stages],
    }
    if passband is not None and stopband is not None:
        result["metrics"] = compute_metrics(stages, passband, stopband, grid_points)
    return result
