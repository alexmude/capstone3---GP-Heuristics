from copy import deepcopy
from typing import Dict, Optional
from csp.csp_instance import CSPInstance
from heuristics.base import VariableHeuristic
from solver.ac3 import ac3


class BacktrackingSolver:
    def __init__(self, use_ac3: bool = True) -> None:
        self.use_ac3 = use_ac3
        self.metrics = {
            "consistency_checks": 0,
            "backtracks": 0,
            "nodes": 0,
        }

    def solve(self, csp: CSPInstance, heuristic: VariableHeuristic) -> Optional[Dict[str, int]]:
        self.metrics = {"consistency_checks": 0, "backtracks": 0, "nodes": 0}
        domains = {v: set(vals) for v, vals in csp.domains.items()}
        if self.use_ac3:
            ok = ac3(csp, domains, self.metrics)
            if not ok:
                return None
        return self._backtrack(csp, {}, domains, heuristic)
    
    def _backtrack(
        self,
        csp: CSPInstance,
        assignment: Dict[str, int],
        domains: Dict[str, set[int]],
        heuristic: VariableHeuristic,
    ) -> Optional[Dict[str, int]]:
        if len(assignment) == len(csp.variables):
            return assignment.copy()

        self.metrics["nodes"] += 1
        var = self._select_unassigned_variable(csp, assignment, domains, heuristic)

        for value in sorted(domains[var]):
            if self._is_consistent(csp, var, value, assignment):
                new_assignment = assignment.copy()
                new_assignment[var] = value
                new_domains = deepcopy(domains)
                new_domains[var] = {value}

                valid = True
                if self.use_ac3:
                    valid = ac3(csp, new_domains, self.metrics)

                if valid:
                    result = self._backtrack(csp, new_assignment, new_domains, heuristic)
                    if result is not None:
                        return result

        self.metrics["backtracks"] += 1
        return None
    
    def _select_unassigned_variable(
        self,
        csp: CSPInstance,
        assignment: Dict[str, int],
        domains: Dict[str, set[int]],
        heuristic: VariableHeuristic,
    ) -> str:
        candidates = [v for v in csp.variables if v not in assignment]
        return min(candidates, key=lambda v: (heuristic.score(v, csp, domains, assignment), v))

    def _is_consistent(self, csp: CSPInstance, var: str, value: int, assignment: Dict[str, int]) -> bool:
        for other_var, other_value in assignment.items():
            self.metrics["consistency_checks"] += 1
            if not csp.is_consistent_pair(var, value, other_var, other_value):
                return False
        return True