import random
from typing import Dict, List, Set, Tuple
from csp.csp_instance import CSPInstance


def generate_random_binary_csp(
    n_vars: int,
    domain_size: int,
    density: float,
    tightness: float,
    seed: int | None = None,
) -> CSPInstance:
    # Initialize random generator (for reproducibility if seed is provided)
    rng = random.Random(seed)

    # Create variable names: X0, X1, ..., X(n-1)
    variables = [f"X{i}" for i in range(n_vars)]

    # Assign each variable a domain: {0, 1, ..., domain_size-1}
    domains: Dict[str, Set[int]] = {
        v: set(range(domain_size)) for v in variables
    }

    # Dictionary to store constraints:
    # key = (variable_i, variable_j)
    # value = set of allowed (value_i, value_j) pairs
    constraints: Dict[Tuple[str, str], Set[Tuple[int, int]]] = {}

    
    # STEP 1: Generate all possible variable pairs
    
    # Example: (X0, X1), (X0, X2), ..., (X1, X2), ...
    all_pairs = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            all_pairs.append((variables[i], variables[j]))

 
    # STEP 2: Select subset of pairs based on density
  
    # density controls how many constraints exist
    # higher density = more constrained variable pairs
    num_edges = max(1, int(density * len(all_pairs)))
    chosen_edges = rng.sample(all_pairs, num_edges)

   
    # STEP 3: For each chosen pair, define constraints
    
    for (x, y) in chosen_edges:

        # Generate all possible value combinations between x and y
        # Example: if domain = {0,1,2}, then pairs = (0,0), (0,1), ...
        all_value_pairs = [(a, b) for a in domains[x] for b in domains[y]]

        # STEP 4: Apply tightness
    
        # tightness controls how restrictive constraints are
        # it defines how many pairs are FORBIDDEN
        num_forbidden = int(tightness * len(all_value_pairs))

        # Randomly select forbidden pairs
        forbidden = (
            set(rng.sample(all_value_pairs, num_forbidden))
            if num_forbidden > 0
            else set()
        )

        # Allowed pairs = all pairs minus forbidden ones
        allowed = set(all_value_pairs) - forbidden

        # Store allowed pairs as the constraint
        constraints[(x, y)] = allowed

    # STEP 5: Return CSP instance
   
    return CSPInstance(
        variables=variables,
        domains=domains,
        constraints=constraints
    )