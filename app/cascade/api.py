"""级联链路接口路由：与单滤波器接口并列的独立处理路径。

POST /api/v1/cascade/analyze
    按顺序给一串滤波段（系数段或设计意图段），合成等效系统并
    计算链路级指标（通带纹波 / 阻带衰减 / 相位线性度与群延时 /
    稳定性归因）。等效系数格式与单滤波器接口一致，可直接喂回
    /api/v1/filters/frequency 与 /api/v1/filters/zero-poles 复核。
"""

from __future__ import annotations

from fastapi import APIRouter

from .schemas import CascadeRequest
from .service import analyze_cascade

router = APIRouter(prefix="/api/v1/cascade", tags=["cascade"])


@router.post("/analyze")
async def analyze(request: CascadeRequest) -> dict:
    return analyze_cascade(request)
