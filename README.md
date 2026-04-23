# Genetic Programming Hyper-Heuristic for CSPs

## How to Run

1. Make sure Python is installed:
   python --version

2. Run the project:
   python main.py



##  What Happens

- Generates CSP datasets (Class A and Class B)
- Runs baseline heuristics (DOM, DEG, MXC, KAPPA)
- Evolves new heuristics using Genetic Programming
- Prints performance results

---

##  Structure

- csp/ → problem + generator
- solver/ → backtracking + AC3
- heuristics/ → standard + GP
- gp/ → evolution engine
- experiments/ → runner
- main.py → entry point

---

## Config

Edit:
- experiments/runner.py → dataset
- gp/engine.py → GP parameters

---

## Run

python main.py
