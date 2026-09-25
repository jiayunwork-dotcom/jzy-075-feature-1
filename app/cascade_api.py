"""级联链路 HTTP 路由：独立路径，不与单滤波器接口混在一起。

POST /api/v1/filters/cascade
    按顺序吃下一串滤波段（设计意图或已落地系数），合成等效单一系统：
    返回与单滤波器接口同格式的等效 b/a（可直接喂回频响、零极点接口），
    回显各段落地后的系数与次序，并基于整条链路给出通带纹波、阻带衰减、
    相位线性度 / 群延时与稳定性归属。
"""

from __future__ import annotations

from fastapi import APIRouter

from .cascade import cascade_coefficients, resolve_segments, stability_report
from .chain_metrics import compute_chain_metrics, validate_band
from .schemas import CascadeRequest, CascadeResponse

router = APIRouter()


@router.post("/api/v1/filters/cascade", response_model=CascadeResponse)
async def cascade_filter(request: CascadeRequest) -> CascadeResponse:
    segments = resolve_segments(request.segments)
    passband = validate_band(request.passband, "通带 passband")
    stopband = validate_band(request.stopband, "阻带 stopband")

    # 等效系数：分子、分母分别卷积，格式与单滤波器设计结果一致
    numerator, denominator = cascade_coefficients(segments)

    # 指标与稳定性：逐段求值 / 逐段求根，高阶链路下数值依然稳
    poles_report = stability_report(segments)
    metrics = compute_chain_metrics(segments, passband, stopband, poles_report)

    return CascadeResponse(
        b=numerator,
        a=denominator,
        segments=[
            {
                "position": segment.position,
                "type": segment.kind,
                "b": segment.b,
                "a": segment.a,
                "spec": segment.spec,
            }
            for segment in segments
        ],
        metrics=metrics,
    )
