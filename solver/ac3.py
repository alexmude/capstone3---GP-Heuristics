from collections import deque
from typing import Dict, Tuple
from csp.csp_instance import CSPInstance


def revise(csp: CSPInstance, xi: str, xj: str, domains: Dict[str, set[int]], metrics: dict) -> bool:
    revised = False
    to_remove = set()
    for vi in domains[xi]:
        supported = False
        for vj in domains[xj]:
            metrics["consistency_checks"] += 1
            if csp.is_consistent_pair(xi, vi, xj, vj):
                supported = True
                break
        if not supported:
            to_remove.add(vi)
    if to_remove:
        domains[xi] -= to_remove
        revised = True
    return revised

def ac3(csp: CSPInstance, domains: Dict[str, set[int]], metrics: dict) -> bool:
    queue = deque((xi, xj) for (xi, xj) in csp.constraints.keys())
    while queue:
        xi, xj = queue.popleft()
        if revise(csp, xi, xj, domains, metrics):
            if not domains[xi]:
                return False
            for xk in csp.neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True

