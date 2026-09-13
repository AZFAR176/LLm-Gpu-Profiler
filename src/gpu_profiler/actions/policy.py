from __future__ import annotations

from gpu_profiler.actions.models import PolicyDecision, TypedAction


class ActionPolicyValidator:
    """Safety and policy checks for typed profiling actions."""

    def validate(self, action: TypedAction) -> PolicyDecision:
        steps = action.params.get("steps")
        if steps is not None:
            if not isinstance(steps, int) or steps <= 0 or steps > 1000:
                return PolicyDecision(
                    allowed=False,
                    message="Invalid 'steps' parameter; allowed range is 1..1000.",
                )
            if steps > 500:
                return PolicyDecision(
                    allowed=False,
                    message="Steps > 500 require human approval in this policy profile.",
                    requires_human_approval=True,
                )

        return PolicyDecision(allowed=True, message="Action allowed by policy.")
