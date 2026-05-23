"""Plugin-style guardrail registry for staged tinyctx checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


GuardFn = Callable[[Mapping[str, Any]], "GuardCheckResult"]


@dataclass(frozen=True)
class GuardCheckResult:
    name: str
    passed: bool
    action: str = "allow"
    reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardrailPlugin:
    name: str
    stage: str
    check: GuardFn
    description: str = ""


class GuardrailRegistry:
    def __init__(self) -> None:
        self._plugins: list[GuardrailPlugin] = []

    def register(self, plugin: GuardrailPlugin) -> None:
        if any(existing.name == plugin.name for existing in self._plugins):
            raise ValueError(f"guardrail already registered: {plugin.name}")
        self._plugins.append(plugin)

    def plugins(self, stage: Optional[str] = None) -> list[GuardrailPlugin]:
        if stage is None:
            return list(self._plugins)
        return [plugin for plugin in self._plugins if plugin.stage == stage]

    def run(
        self,
        context: Mapping[str, Any],
        *,
        stage: Optional[str] = None,
    ) -> list[GuardCheckResult]:
        results: list[GuardCheckResult] = []
        for plugin in self.plugins(stage):
            try:
                result = plugin.check(context)
            except Exception as exc:  # noqa: BLE001 - guard failures must be data
                result = GuardCheckResult(
                    name=plugin.name,
                    passed=False,
                    action="block",
                    reason=str(exc),
                )
            results.append(result)
        return results


def summarize_results(results: list[GuardCheckResult]) -> dict[str, Any]:
    failed = [result for result in results if not result.passed]
    actions: dict[str, int] = {}
    for result in results:
        actions[result.action] = actions.get(result.action, 0) + 1
    return {
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "actions": actions,
        "blocking": [result.name for result in failed if result.action == "block"],
    }
