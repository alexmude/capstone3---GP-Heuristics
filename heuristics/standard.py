from typing import Dict
from heuristics.base import VariableHeuristic
from csp.csp_instance import CSPInstance
from csp.features import degree, domain_size, conflicts, kappa


# DOM = "Minimum Remaining Values" (choose smallest domain first)
class DOMHeuristic(VariableHeuristic):
    def score(self, var: str, csp: CSPInstance, domains: Dict[str, set[int]], assignment: Dict[str, int]) -> float:
        # Return the size of the domain
        # Smaller domain → lower score → chosen first
        return float(domain_size(var, domains))


# DEG = "Maximum Degree" (choose variable connected to many others)
class DEGHeuristic(VariableHeuristic):
    def score(self, var: str, csp: CSPInstance, domains: Dict[str, set[int]], assignment: Dict[str, int]) -> float:
        # Degree = number of neighbors not yet assigned
        # We negate it because solver picks MIN score
        # More neighbors: larger degree: more negative: chosen first
        return float(-degree(var, csp, assignment))


# MXC = "Maximum Conflicts" (choose variable involved in many conflicts)
class MXCHeuristic(VariableHeuristic):
    def score(self, var: str, csp: CSPInstance, domains: Dict[str, set[int]], assignment: Dict[str, int]) -> float:
        # Count how many value pairs violate constraints
        # More conflicts: more constrained: should be chosen earlier
        # Negate because solver selects minimum score
        return float(-conflicts(var, csp, domains, assignment))


# KAPPA = Constraint tightness-based heuristic
class KAPPAHeuristic(VariableHeuristic):
    def score(self, var: str, csp: CSPInstance, domains: Dict[str, set[int]], assignment: Dict[str, int]) -> float:
        # Kappa measures how restrictive the constraints are
        # Higher kappa → tighter constraints → more important variable
        # Negate because solver picks minimum score
        return float(-kappa(var, csp, domains, assignment))