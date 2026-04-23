import math
from typing import Dict
from csp.csp_instance import CSPInstance



# FEATURE 1: DEGREE

def degree(var: str, csp: CSPInstance, assignment: Dict[str, int]) -> int:
    """
    Returns the number of neighbors of 'var' that are NOT yet assigned.

    Intuition:
    Measures how "connected" a variable is
    Higher degree = more constraints with other variables
    Important for DEG heuristic (choose most constrained variable structurally)
    """
    return sum(1 for n in csp.neighbors[var] if n not in assignment)


# FEATURE 2: DOMAIN SIZE
def domain_size(var: str, domains: Dict[str, set[int]]) -> int:
    """
    Returns the number of remaining possible values for 'var'.

    Intuition:
    Smaller domain = more constrained variable
    Used in DOM heuristic (choose variable with smallest domain)
    """
    return len(domains[var])

# FEATURE 3: CONFLICTS

def conflicts(var: str, csp: CSPInstance, domains: Dict[str, set[int]], assignment: Dict[str, int]) -> int:
    """
    Counts how many value pairs involving 'var' would violate constraints.

    For each neighbor:
    Try all combinations of values (a, b)
    Count how many pairs are inconsistent

    Intuition:
    Higher conflicts = more restrictive variable
    Used in MXC-like reasoning (focus on problematic variables)
    """
    total = 0
    for n in csp.neighbors[var]:
        if n in assignment:
            continue
        for a in domains[var]:
            for b in domains[n]:
                # If pair is NOT allowed → conflict
                if not csp.is_consistent_pair(var, a, n, b):
                    total += 1
    return total

# FEATURE 4: KAPPA (constraint tightness measure)

def kappa(var: str, csp: CSPInstance, domains: Dict[str, set[int]], assignment: Dict[str, int]) -> float:
    """
    Computes a measure of constraint tightness around a variable.

    Steps:
    1. For each neighbor:
    Count how many value pairs are forbidden
    Compute probability of conflict (p_c)
    2. Apply information-theoretic transformation using log

    Intuition:
    Measures how "restrictive" the constraints are
    Higher kappa = tighter constraints
    More informative than raw conflict counts
    """
    dom = max(1, len(domains[var]))  
    numerator = 0.0

    for n in csp.neighbors[var]:
        if n in assignment:
            continue

        total_pairs = len(domains[var]) * len(domains[n])
        if total_pairs == 0:
            continue

        forbidden = 0

        # Count forbidden pairs
        for a in domains[var]:
            for b in domains[n]:
                if not csp.is_consistent_pair(var, a, n, b):
                    forbidden += 1

        # Probability of conflict
        p_c = forbidden / total_pairs

        # Avoid log(0)
        if p_c < 1.0:
            numerator -= math.log2(max(1e-9, 1 - p_c))

    # Normalize by domain size
    return numerator / max(1e-9, math.log2(max(2, dom)))


# NORMALIZATION FUNCTION

def normalize_feature_map(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Normalizes feature values into range [0, 1].

    Why normalize?
    - Different features have different scales
    - Ensures fair combination in GP expressions
    - Prevents one feature from dominating

    Edge case:
    - If all values are equal → return 0 for all
    """
    values = list(raw.values())
    vmin, vmax = min(values), max(values)

    if abs(vmax - vmin) < 1e-9:
        return {k: 0.0 for k in raw}

    return {k: (v - vmin) / (vmax - vmin) for k, v in raw.items()}



# MAIN FEATURE COMPUTATION

def compute_all_features(csp: CSPInstance, domains: Dict[str, set[int]], assignment: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Computes all features for all unassigned variables.

    Steps:
    1. Identify unassigned variables
    2. Compute raw features:
        domain size
        degree
        conflicts
        kappa
    3. Normalize each feature across variables
    4. Return dictionary:

        {
            var1: { "dom": ..., "deg": ..., "conflicts": ..., "kappa": ... },
            var2: { ... },
            ...
        }

    Why this matters:
    This is the input to all heuristics (standard + GP)
    GP builds expressions using these normalized features
    """
    # Only consider variables not yet assigned
    unassigned = [v for v in csp.variables if v not in assignment]

    # Compute raw features
    raw_dom = {v: float(domain_size(v, domains)) for v in unassigned}
    raw_deg = {v: float(degree(v, csp, assignment)) for v in unassigned}
    raw_conf = {v: float(conflicts(v, csp, domains, assignment)) for v in unassigned}
    raw_kappa = {v: float(kappa(v, csp, domains, assignment)) for v in unassigned}

    # Normalize features
    norm_dom = normalize_feature_map(raw_dom)
    norm_deg = normalize_feature_map(raw_deg)
    norm_conf = normalize_feature_map(raw_conf)
    norm_kappa = normalize_feature_map(raw_kappa)

    # Combine into final structure
    out: Dict[str, Dict[str, float]] = {}
    for v in unassigned:
        out[v] = {
            "dom": norm_dom[v],
            "deg": norm_deg[v],
            "conflicts": norm_conf[v],
            "kappa": norm_kappa[v],
        }

    return out