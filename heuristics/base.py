from abc import ABC, abstractmethod
from typing import Dict
from csp.csp_instance import CSPInstance


# Base class for all variable-ordering heuristics
# Every heuristic (DOM, DEG, GP, etc.) must inherit from this
class VariableHeuristic(ABC):

    # This method must be implemented by all subclasses
    # It assigns a numeric score to a variable
    @abstractmethod
    def score(
        self,
        var: str,                         # variable being evaluated
        csp: CSPInstance,                 # the CSP problem
        domains: Dict[str, set[int]],     # current domains of all variables
        assignment: Dict[str, int]        # current partial assignment
    ) -> float:
        
        # The solver will use this score to decide which variable to pick next
        # Lower score = higher priority (because we use min(...) in the solver)
        
        raise NotImplementedError