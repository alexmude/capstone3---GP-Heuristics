from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Optional

TERMINALS = ["dom", "deg", "conflicts", "kappa", "r"]
FUNCTIONS = ["+", "-", "*", "/"]


@dataclass
class GPNode:
    value: str
    left: Optional["GPNode"] = None
    right: Optional["GPNode"] = None

    def is_terminal(self) -> bool:
        return self.value in TERMINALS or self.value.replace(".", "", 1).replace("-", "", 1).isdigit()

    def evaluate(self, features: dict[str, float]) -> float:
        if self.value in TERMINALS:
            return features[self.value]
        if self.is_terminal():
            return float(self.value)

        left_val = self.left.evaluate(features)
        right_val = self.right.evaluate(features)

        if self.value == "+":
            return left_val + right_val
        if self.value == "-":
            return left_val - right_val
        if self.value == "*":
            return left_val * right_val
        if self.value == "/":
            return 1.0 if abs(right_val) < 1e-9 else left_val / right_val

        raise ValueError(f"Unknown node value: {self.value}")

    def clone(self) -> "GPNode":
        return GPNode(
            self.value,
            self.left.clone() if self.left else None,
            self.right.clone() if self.right else None,
        )

    def to_string(self) -> str:
        if self.left is None and self.right is None:
            return self.value
        return f"({self.left.to_string()} {self.value} {self.right.to_string()})"


def random_tree(max_depth: int, rng: random.Random, force_function: bool = False) -> GPNode:
    if max_depth == 0 and not force_function:
        return GPNode(rng.choice(TERMINALS))

    if max_depth == 0:
        op = rng.choice(FUNCTIONS)
        return GPNode(op, GPNode(rng.choice(TERMINALS)), GPNode(rng.choice(TERMINALS)))

    if force_function or rng.random() < 0.7:
        op = rng.choice(FUNCTIONS)
        return GPNode(
            op,
            random_tree(max_depth - 1, rng),
            random_tree(max_depth - 1, rng),
        )

    return GPNode(rng.choice(TERMINALS))