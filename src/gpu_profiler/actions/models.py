from __future__ import annotations

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    COLLECT_NCCL_BREAKDOWN = "collect_nccl_breakdown_100_steps"
    COLLECT_GPU_IDLE_TIMELINE = "collect_gpu_idle_timeline_100_steps"
    COLLECT_NCCL_COLLECTIVE_BREAKDOWN = "collect_nccl_collective_breakdown_100_steps"
    COLLECT_CPU_STAGE_BREAKDOWN = "collect_cpu_stage_breakdown_100_steps"
    COLLECT_PHASE_SEGMENTED_TIMELINE = "collect_phase_segmented_timeline_200_steps"


class TypedAction(BaseModel):
    action_type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)
    reason: str


class PolicyDecision(BaseModel):
    allowed: bool
    message: str
    requires_human_approval: bool = False
