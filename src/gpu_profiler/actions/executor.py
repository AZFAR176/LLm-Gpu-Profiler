from __future__ import annotations

import json
from pathlib import Path

from gpu_profiler.actions.models import ActionType, TypedAction
from gpu_profiler.actions.policy import ActionPolicyValidator
from gpu_profiler.models import ProfilingActionRequest, ProfilingActionResult, TelemetryFrame


ALLOWLIST = {action.value for action in ActionType}


class ProfilingActionExecutor:
    """
    Executes profiling requests by loading action-specific telemetry artifacts.

    Expected artifact path:
      <followup_path>/actions/<action_id>.json
    """

    def __init__(self) -> None:
        self.policy = ActionPolicyValidator()

    def execute(
        self, request: ProfilingActionRequest, followup_path: Path | None
    ) -> tuple[ProfilingActionResult, TelemetryFrame | None]:
        typed_or_error = self._to_typed_action(request)
        if isinstance(typed_or_error, str):
            return (
                ProfilingActionResult(
                    action_id=request.action_id,
                    status="blocked_unsafe",
                    details=typed_or_error,
                ),
                None,
            )

        decision = self.policy.validate(typed_or_error)
        if not decision.allowed:
            status = "blocked_needs_approval" if decision.requires_human_approval else "blocked_unsafe"
            return (
                ProfilingActionResult(
                    action_id=request.action_id,
                    status=status,
                    details=decision.message,
                ),
                None,
            )

        if followup_path is None:
            return (
                ProfilingActionResult(
                    action_id=request.action_id,
                    status="skipped",
                    details="No followup path configured for action execution.",
                ),
                None,
            )

        action_file = followup_path / "actions" / f"{request.action_id}.json"
        if not action_file.exists():
            return (
                ProfilingActionResult(
                    action_id=request.action_id,
                    status="missing_artifact",
                    details=f"Expected action telemetry not found: {action_file}",
                ),
                None,
            )

        payload = json.loads(action_file.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        metadata = payload.get("metadata", {})
        metadata["action_id"] = request.action_id
        metadata["action_reason"] = request.reason

        frame = TelemetryFrame(metrics=metrics, metadata=metadata)
        return (
            ProfilingActionResult(
                action_id=request.action_id,
                status="executed",
                details=f"Loaded followup metrics from {action_file}",
            ),
            frame,
        )

    def _to_typed_action(self, request: ProfilingActionRequest) -> TypedAction | str:
        if request.action_id not in ALLOWLIST:
            return f"Action '{request.action_id}' is not allowlisted."
        try:
            action_type = ActionType(request.action_id)
        except Exception:
            return f"Action '{request.action_id}' could not be converted to typed action."
        return TypedAction(action_type=action_type, params=request.params, reason=request.reason)
