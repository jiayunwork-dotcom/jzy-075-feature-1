"""级联链路接口的请求模型。

与单滤波器接口同一风格：模型层只做 JSON 结构解析，业务规则
（段描述、频带、采样点数）由 cascade.service 显式校验，
保证非法输入一律拿到带中文原因的 400。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class CascadeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    segments: Any = None  # 段描述数组；类型与内容由 service 校验
    passband: Any = None  # [下限, 上限]，相对奈奎斯特，可缺省（不算指标）
    stopband: Any = None
    grid_points: Any = None  # 指标采样点数（整数 32..8192，缺省 4096）
