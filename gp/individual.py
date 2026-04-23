from dataclasses import dataclass, field
from gp.tree import GPNode


@dataclass
class GPIndividual:
    tree: GPNode
    fitness: float = field(default=0.0)

    def evaluate(self, features: dict[str, float]) -> float:
        return self.tree.evaluate(features)

    def clone(self) -> "GPIndividual":
        return GPIndividual(self.tree.clone(), self.fitness)