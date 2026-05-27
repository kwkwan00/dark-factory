"""Role-specialized agents for the adversarial refinery.

Information-hiding boundary (Parnas): the orchestrator imports only the
RoleAgent ABC and the RoleRegistry from this package. Concrete role
classes (ProductRole, EngineeringRole, etc.) must not be imported by
anything outside this package — only the registry instantiates them.
"""

from dark_factory.api.refinery.roles.base import RoleAgent
from dark_factory.api.refinery.roles.registry import RoleRegistry

__all__ = ["RoleAgent", "RoleRegistry"]
