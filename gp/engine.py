import random
from gp.tree import random_tree
from gp.individual import GPIndividual
from gp.operators import crossover, mutate
from heuristics.gp_heuristic import GPHeuristic
from solver.backtracking_solver import BacktrackingSolver


class GPEngine:
    def __init__(
        self,
        population_size: int = 20,
        generations: int = 12,
        max_depth: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)

    def initialize_population(self) -> list[GPIndividual]:
        return [
            GPIndividual(random_tree(self.max_depth, self.rng, force_function=True))
            for _ in range(self.population_size)
        ]

    def evaluate_population_fitness(
        self,
        population: list[GPIndividual],
        instances: list,
    ) -> None:
        """
        Paper-style normalized fitness:
            q(i,j) = 1 / cost(i,j)
            q_hat(i,j) = q(i,j) / max_k q(k,j)
            f(i) = sum_j q_hat(i,j)
        """
        num_individuals = len(population)
        num_instances = len(instances)

        # costs[i][j] = consistency checks of individual i on instance j
        costs = [[0.0 for _ in range(num_instances)] for _ in range(num_individuals)]

        for i, individual in enumerate(population):
            for j, csp in enumerate(instances):
                heuristic = GPHeuristic(individual, seed=j)
                solver = BacktrackingSolver(use_ac3=True)
                solver.solve(csp, heuristic)

                cost = solver.metrics["consistency_checks"]

                # protect against zero cost
                if cost <= 0:
                    cost = 1

                costs[i][j] = float(cost)

        # Convert to qualities q(i,j) = 1 / cost(i,j)
        qualities = [[1.0 / costs[i][j] for j in range(num_instances)] for i in range(num_individuals)]

        # Normalize per instance and sum
        for i in range(num_individuals):
            population[i].fitness = 0.0

        for j in range(num_instances):
            best_quality_for_instance = max(qualities[i][j] for i in range(num_individuals))

            # extra guard, though it should never be zero
            if best_quality_for_instance <= 0:
                best_quality_for_instance = 1e-9

            for i in range(num_individuals):
                normalized_quality = qualities[i][j] / best_quality_for_instance
                population[i].fitness += normalized_quality

    def tournament(self, population: list[GPIndividual], k: int = 2) -> GPIndividual:
        sample = self.rng.sample(population, k)
        return max(sample, key=lambda ind: ind.fitness)

    def evolve(self, instances: list) -> GPIndividual:
        population = self.initialize_population()
        best = None

        for gen in range(self.generations):
            self.evaluate_population_fitness(population, instances)

            generation_best = max(population, key=lambda ind: ind.fitness)

            if best is None or generation_best.fitness > best.fitness:
                best = generation_best.clone()

            new_population = [best.clone()]  # elitism

            while len(new_population) < self.population_size:
                p1 = self.tournament(population)
                p2 = self.tournament(population)

                if self.rng.random() < self.crossover_rate:
                    c1, c2 = crossover(p1, p2, self.rng)
                else:
                    c1, c2 = p1.clone(), p2.clone()

                if self.rng.random() < self.mutation_rate:
                    c1 = mutate(c1, self.rng, self.max_depth)
                if self.rng.random() < self.mutation_rate:
                    c2 = mutate(c2, self.rng, self.max_depth)

                new_population.extend([c1, c2])

            population = new_population[: self.population_size]

            print(
                f"Generation {gen + 1}: "
                f"best fitness={best.fitness:.6f}, "
                f"heuristic={best.tree.to_string()}"
            )

        return best