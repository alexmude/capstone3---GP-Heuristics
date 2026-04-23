from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any

@dataclass
class CSPInstance:
    # List of variable names in the CSP
    variables: List[str]

    # Domain of each variable: variable -> set of possible values
    # Example: {"X1": {1,2,3}, "X2": {1,2,3}}
    domains: Dict[str, Set[int]]

    # Constraints stored as allowed value pairs between variable pairs
    # Key: (xi, xj)
    # Value: set of allowed tuples (vi, vj)
    # Example: ("X1","X2") -> {(1,2), (2,3)}
    constraints: Dict[Tuple[str, str], Set[Tuple[int, int]]]

    # Neighbor list: for each variable, which other variables it is constrained with
    # This is computed automatically after initialization
    neighbors: Dict[str, Set[str]] = field(init=False)

    def __post_init__(self) -> None:
        """
        Automatically build the neighbor structure after the object is created.

        neighbors[x] = set of variables that have a constraint with x

        This is used heavily in:
        - AC-3 
        - feature computation (degree, conflicts)
        """
        # Initialize empty neighbor sets for each variable
        self.neighbors = {var: set() for var in self.variables}

        # For every constraint (x, y), add each variable as neighbor of the other
        for (x, y) in self.constraints:
            self.neighbors[x].add(y)
            self.neighbors[y].add(x)

    def is_consistent_pair(self, xi: str, vi: int, xj: str, vj: int) -> bool:
        """
        Check whether assigning xi = vi and xj = vj is consistent with constraints.

        Returns:
            True: assignment is allowed (no violation)
            False: assignment violates a constraint

        This function is used in:
        - backtracking solver (to reject invalid assignments)
        - AC-3 (to remove inconsistent domain values)
        - heuristic features (conflict counting)
        """

        # 1: constraint exists in forward direction (xi -> xj)
        if (xi, xj) in self.constraints:
            # Check if the pair (vi, vj) is allowed
            # If not in the allowed set → violation
            return (vi, vj) in self.constraints[(xi, xj)]

        # 2: constraint exists in reverse direction (xj -> xi)
        if (xj, xi) in self.constraints:
            # Since constraint is reversed, we must flip the values
            # (vj, vi) must be checked instead of (vi, vj)
            return (vj, vi) in self.constraints[(xj, xi)]

        # 3: no constraint exists between xi and xj
        return True