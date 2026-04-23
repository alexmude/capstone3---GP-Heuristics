from csp.generator import generate_random_binary_csp
from heuristics.standard import DOMHeuristic, DEGHeuristic, KAPPAHeuristic, MXCHeuristic
from heuristics.gp_heuristic import GPHeuristic
from solver.backtracking_solver import BacktrackingSolver
from gp.engine import GPEngine


# Evaluate a heuristic on a dataset

def evaluate_heuristic(instances, heuristic, name: str):
    """
    Runs a given heuristic on a list of CSP instances and reports:
    number of solved instances
    average consistency checks (main cost metric)
    average number of backtracks
    """
    total_checks = 0
    total_backtracks = 0
    solved = 0

    for csp in instances:
        # Create solver with AC-3 enabled (constraint propagation)
        solver = BacktrackingSolver(use_ac3=True)

        # Solve the CSP using the given heuristic
        result = solver.solve(csp, heuristic)

        # Count how many instances were solved
        if result is not None:
            solved += 1

        # Accumulate performance metrics
        total_checks += solver.metrics["consistency_checks"]
        total_backtracks += solver.metrics["backtracks"]

    # Print summary statistics
    print(
        f"{name}: solved={solved}/{len(instances)}, "
        f"avg_checks={total_checks / len(instances):.2f}, "
        f"avg_backtracks={total_backtracks / len(instances):.2f}"
    )


# Dataset generator helper

def make_dataset(n_instances: int, n_vars: int, domain_size: int, density: float, tightness: float, start_seed: int):
    """
    Generates a list of random CSP instances.

    Parameters:
    n_instances: number of CSPs to generate
    n_vars: number of variables in each CSP
    domain_size: number of possible values per variable
    density: probability that a constraint exists between two variables
    tightness: fraction of value pairs that are forbidden (constraint strength)
    start_seed: ensures reproducibility across runs

    Each instance uses a different seed for diversity.
    """
    return [
        generate_random_binary_csp(
            n_vars=n_vars,
            domain_size=domain_size,
            density=density,
            tightness=tightness,
            seed=start_seed + i,  # different seed for each instance
        )
        for i in range(n_instances)
    ]


# Run one experiment (train GP + evaluate)

def run_one_experiment(label: str, train, test):
    """
    Runs a full experiment:
    1. Evaluate baseline heuristics on training data
    2. Train GP heuristic on training data
    3. Evaluate all heuristics on test data
    """

    
    print(f"Experiment: {label}")
    


    # Step 1: Baseline comparison
 
    print("=== Baselines on training ===")
    evaluate_heuristic(train, DOMHeuristic(), "DOM")
    evaluate_heuristic(train, DEGHeuristic(), "DEG")
    evaluate_heuristic(train, MXCHeuristic(), "MXC")
    evaluate_heuristic(train, KAPPAHeuristic(), "KAPPA")

    # Step 2: Train GP heuristic
  
    print("=== Evolving GP heuristic ===")

    # Configure Genetic Programming engine
    engine = GPEngine(
        population_size=16,   # number of heuristics per generation
        generations=8,        # number of evolution steps
        max_depth=3,          # max complexity of expression trees
        crossover_rate=0.9,   # probability of combining two heuristics
        mutation_rate=0.1,    # probability of random modification
        seed=42,              # reproducibility
    )

    # Run evolution on training set
    best = engine.evolve(train)

    # Wrap evolved tree into a usable heuristic
    gp_heuristic = GPHeuristic(best, seed=999)

    # Step 3: Evaluate on test set
   
    print("=== Evaluation on test ===")
    evaluate_heuristic(test, DOMHeuristic(), "DOM")
    evaluate_heuristic(test, DEGHeuristic(), "DEG")
    evaluate_heuristic(test, MXCHeuristic(), "MXC")
    evaluate_heuristic(test, KAPPAHeuristic(), "KAPPA")

    # Evaluate evolved GP heuristic
    evaluate_heuristic(test, gp_heuristic, f"GP {best.tree.to_string()}")


# Main experiment pipeline

def run_experiment():
    """
    Runs two experiments on different CSP classes:

    Class A:
    - Sparse constraint graph (low density)
    - Weak constraints (low tightness)
    - Easier problems → DOM usually best

    Class B:
    - Denser constraint graph (higher density)
    - Stronger constraints (higher tightness)
    - Harder problems → conflict-based heuristics (MXC) perform better
    """

    # Class A: Sparse / Low Tightness
   
    train_a = make_dataset(
        n_instances=15,
        n_vars=10,
        domain_size=5,
        density=0.25,      # few constraints → sparse graph
        tightness=0.25,    # many allowed value pairs: loose constraints
        start_seed=0,
    )

    test_a = make_dataset(
        n_instances=30,
        n_vars=10,
        domain_size=5,
        density=0.25,
        tightness=0.25,
        start_seed=1000,
    )

    # Class B: Denser / Higher Tightness
    
    train_b = make_dataset(
        n_instances=15,
        n_vars=10,
        domain_size=5,
        density=0.55,      # more constraints → denser graph
        tightness=0.45,    # more forbidden pairs → stronger constraints
        start_seed=2000,
    )

    test_b = make_dataset(
        n_instances=30,
        n_vars=10,
        domain_size=5,
        density=0.55,
        tightness=0.45,
        start_seed=3000,
    )

    # Run experiments on both classes
    run_one_experiment("Class A (sparse / low tightness)", train_a, test_a)
    run_one_experiment("Class B (denser / higher tightness)", train_b, test_b)