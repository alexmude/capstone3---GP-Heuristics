import random
from gp.tree import GPNode, random_tree
from gp.individual import GPIndividual


def collect_nodes(root: GPNode):
    nodes = [root]
    if root.left:
        nodes.extend(collect_nodes(root.left))
    if root.right:
        nodes.extend(collect_nodes(root.right))
    return nodes


def mutate(ind: GPIndividual, rng: random.Random, max_depth: int) -> GPIndividual:
    clone = ind.clone()
    nodes = collect_nodes(clone.tree)
    target = rng.choice(nodes)
    replacement = random_tree(max_depth=max_depth, rng=rng)
    target.value = replacement.value
    target.left = replacement.left
    target.right = replacement.right
    return clone

def crossover(a: GPIndividual, b: GPIndividual, rng: random.Random) -> tuple[GPIndividual, GPIndividual]:
    ca, cb = a.clone(), b.clone()
    nodes_a = collect_nodes(ca.tree)
    nodes_b = collect_nodes(cb.tree)
    na = rng.choice(nodes_a)
    nb = rng.choice(nodes_b)
    na.value, nb.value = nb.value, na.value
    na.left, nb.left = nb.left, na.left
    na.right, nb.right = nb.right, na.right
    return ca, cb