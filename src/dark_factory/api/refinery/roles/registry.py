"""Role registry — the only place concrete roles are instantiated.

The orchestrator asks for a role by name; the registry applies per-role
config overrides (model, reasoning effort, filter-policy partial override)
before returning the instance. Tests and plugins can call ``register()``
to swap a role's factory at runtime.
"""

from __future__ import annotations

from collections.abc import Callable

from dark_factory.api.refinery.roles.base import RoleAgent
from dark_factory.config import PipelineConfig
from dark_factory.log import trace_methods


@trace_methods
class RoleRegistry:
    """Look up role instances by name, with config-driven overrides."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._factories: dict[str, Callable[[], RoleAgent]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the built-in roles.

        Imports are deferred so tests can stub roles without pulling
        heavy deps. Each role is best-effort; import failures log a
        warning and skip that registration rather than breaking the
        registry.
        """

        try:
            from dark_factory.api.refinery.roles.product import ProductRole

            self._factories["product"] = ProductRole
        except ImportError:  # pragma: no cover - defensive during migration
            pass
        try:
            from dark_factory.api.refinery.roles.judge import JudgeRole

            self._factories["judge"] = JudgeRole
        except ImportError:  # pragma: no cover - defensive during migration
            pass
        try:
            from dark_factory.api.refinery.roles.engineering import EngineeringRole
            from dark_factory.api.refinery.roles.security import SecurityRole
            from dark_factory.api.refinery.roles.operations import OperationsRole
            from dark_factory.api.refinery.roles.cost import CostRole

            self._factories["engineering"] = EngineeringRole
            self._factories["security"] = SecurityRole
            self._factories["operations"] = OperationsRole
            self._factories["cost"] = CostRole
        except ImportError:  # pragma: no cover - defensive during migration
            pass
    def register(self, name: str, factory: Callable[[], RoleAgent]) -> None:
        """Register or swap a role's factory. Useful for tests."""

        self._factories[name] = factory

    def has(self, name: str) -> bool:
        return name in self._factories

    def get(self, name: str) -> RoleAgent:
        """Return a role instance with per-role config overrides applied."""

        if name not in self._factories:
            raise KeyError(f"role '{name}' is not registered")
        instance = self._factories[name]()
        return instance.configure(
            model=self._config.refinery_role_models.get(name),
            reasoning_effort=self._config.refinery_role_reasoning_effort.get(name),
        )

    def enabled_roles(self) -> list[str]:
        """Return the role names the operator has enabled, filtered to ones
        that have a registered factory."""

        return [n for n in self._config.refinery_roles_enabled if n in self._factories]
