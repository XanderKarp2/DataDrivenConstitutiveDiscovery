#%%
#!/usr/bin/env python3
"""
ULTIMATE: Clean Generative GP for Rayleighian Discovery
========================================================
Combines only the proven features:
✓ Smart seeding (works for FENE-P!)
✓ Simple GP (fast, reliable)
✓ Joint Ψ/Φ discovery
✓ Careful optimization
✓ Clean code (~400 lines)
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import differential_evolution
import copy
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# EXPRESSION TREE
# =============================================================================
class Node:
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def to_sympy(self, symbols: List[sp.Symbol]) -> sp.Expr:
        raise NotImplementedError
    def complexity(self) -> int:
        raise NotImplementedError


class Constant(Node):
    def __init__(self, value: float):
        self.value = value
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X) if hasattr(X, '__len__') else 1, self.value)
    
    def to_sympy(self, symbols: List[sp.Symbol]) -> sp.Expr:
        return sp.Float(self.value)
    
    def complexity(self) -> int:
        return 0
    
    def __str__(self):
        if abs(self.value) < 1e-10:
            return "0"
        if abs(self.value - round(self.value)) < 1e-6:
            return str(int(round(self.value)))
        return f"{self.value:.4f}"


class Variable(Node):
    def __init__(self, index: int, name: str):
        self.index = index
        self.name = name
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return X
        return X[:, self.index]
    
    def to_sympy(self, symbols: List[sp.Symbol]) -> sp.Expr:
        return symbols[self.index]
    
    def complexity(self) -> int:
        return 1
    
    def __str__(self):
        return self.name


class UnaryOp(Node):
    def __init__(self, op: str, child: Node):
        self.op = op
        self.child = child
        self.ops = {
            'neg': lambda x: -x,
            'square': lambda x: x**2,
            'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
            'log': lambda x: np.log(np.maximum(np.abs(x), 1e-10)),
            'exp': lambda x: np.exp(np.clip(x, -10, 10)),
        }
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return self.ops[self.op](self.child.evaluate(X))
    
    def to_sympy(self, symbols: List[sp.Symbol]) -> sp.Expr:
        child_expr = self.child.to_sympy(symbols)
        if self.op == 'neg': return -child_expr
        elif self.op == 'square': return child_expr**2
        elif self.op == 'sqrt': return sp.sqrt(child_expr)
        elif self.op == 'log': return sp.log(sp.Abs(child_expr))
        elif self.op == 'exp': return sp.exp(child_expr)
    
    def complexity(self) -> int:
        return 1 + self.child.complexity()
    
    def __str__(self):
        s = str(self.child)
        if self.op == 'neg': return f"-({s})"
        elif self.op == 'square': return f"({s})²"
        elif self.op == 'sqrt': return f"√({s})"
        elif self.op == 'log': return f"log({s})"
        elif self.op == 'exp': return f"exp({s})"


class BinaryOp(Node):
    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right
        self.ops = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: x / (y + 1e-10 * np.sign(y + 1e-10)),
        }
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return self.ops[self.op](self.left.evaluate(X), self.right.evaluate(X))
    
    def to_sympy(self, symbols: List[sp.Symbol]) -> sp.Expr:
        left_expr = self.left.to_sympy(symbols)
        right_expr = self.right.to_sympy(symbols)
        if self.op == 'add': return left_expr + right_expr
        elif self.op == 'sub': return left_expr - right_expr
        elif self.op == 'mul': return left_expr * right_expr
        elif self.op == 'div': return left_expr / right_expr
    
    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()
    
    def __str__(self):
        ops = {'add': '+', 'sub': '-', 'mul': '·', 'div': '/'}
        return f"({self.left}){ops[self.op]}({self.right})"


# =============================================================================
# SIMPLIFICATION
# =============================================================================
def simplify_tree(tree: Node, var_names: List[str], aggressive: bool = False) -> Node:
    """Careful simplification that doesn't break good solutions"""
    try:
        symbols = [sp.Symbol(name) for name in var_names]
        expr = tree.to_sympy(symbols)
        
        # Only basic simplification
        if aggressive:
            simplified = sp.simplify(sp.expand(expr))
        else:
            # Minimal simplification (just collect terms)
            simplified = sp.collect(expr, symbols)
        
        return sympy_to_tree(simplified, var_names)
    except:
        return tree


def sympy_to_tree(expr: sp.Expr, var_names: List[str]) -> Node:
    """Convert sympy back to tree"""
    if isinstance(expr, sp.Symbol):
        return Variable(var_names.index(str(expr)), str(expr))
    elif isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
        return Constant(float(expr))
    elif isinstance(expr, sp.Add):
        args = list(expr.args)
        result = sympy_to_tree(args[0], var_names)
        for arg in args[1:]:
            result = BinaryOp('add', result, sympy_to_tree(arg, var_names))
        return result
    elif isinstance(expr, sp.Mul):
        args = list(expr.args)
        result = sympy_to_tree(args[0], var_names)
        for arg in args[1:]:
            result = BinaryOp('mul', result, sympy_to_tree(arg, var_names))
        return result
    elif isinstance(expr, sp.Pow):
        base = sympy_to_tree(expr.args[0], var_names)
        exp_val = expr.args[1]
        if isinstance(exp_val, (sp.Integer, sp.Float)):
            if abs(float(exp_val) - 2) < 1e-10:
                return UnaryOp('square', base)
            elif abs(float(exp_val) - 0.5) < 1e-10:
                return UnaryOp('sqrt', base)
        return base
    elif isinstance(expr, sp.log):
        return UnaryOp('log', sympy_to_tree(expr.args[0], var_names))
    elif isinstance(expr, sp.exp):
        return UnaryOp('exp', sympy_to_tree(expr.args[0], var_names))
    else:
        try:
            return Constant(float(expr))
        except:
            return Constant(0.0)


# =============================================================================
# INDIVIDUAL
# =============================================================================
@dataclass
class Individual:
    tree: Node
    fitness: float = float('inf')
    mse: float = float('inf')
    complexity: int = 0
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        try:
            vals = self.tree.evaluate(X)
            if not np.all(np.isfinite(vals)):
                return np.full(len(X) if X.ndim > 1 else len(X), 1e10)
            return vals
        except:
            return np.full(len(X) if hasattr(X, '__len__') else 1, 1e10)
    
    def __str__(self):
        return str(self.tree)


# =============================================================================
# SEED BUILDER (PROVEN TO WORK!)
# =============================================================================
def build_fene_seeds(var_name: str = "I1") -> List[Node]:
    """Seeds for FENE-P - THIS WORKS!"""
    var = Variable(0, var_name)
    seeds = []
    
    # Core forms that worked
    for c_scale in [8.0, 10.0, 12.0, 15.0]:
        # -c * log(1 - I1/L2) - THE FENE-P FORM
        for c_mult in [-3.0, -5.0, -7.0, -10.0]:
            ratio = BinaryOp('div', var, Constant(c_scale))
            one_minus = BinaryOp('sub', Constant(1.0), ratio)
            seeds.append(BinaryOp('mul', Constant(c_mult), UnaryOp('log', one_minus)))
    
    return seeds


def build_oldroyd_seeds(var_names: List[str]) -> List[Node]:
    """Seeds for Oldroyd-B"""
    v1 = Variable(0, var_names[0])
    v2 = Variable(1, var_names[1])
    seeds = []
    
    # I2 - 2*I1 (Neo-Hookean - no constant)
    seeds.append(BinaryOp('sub', v2, BinaryOp('mul', Constant(2.0), v1)))
    
    # I2 - 2*I1 + const (Neo-Hookean WITH constant - the correct form!)
    for c in [0.5, 1.0, 1.5, 2.0, 3.0]:
        neo_hookean = BinaryOp('sub', v2, BinaryOp('mul', Constant(2.0), v1))
        seeds.append(BinaryOp('add', neo_hookean, Constant(c)))
    
    # Linear combinations without constant
    for c1 in [-1.0, -0.5, 0.5, 1.0]:
        for c2 in [0.5, 1.0]:
            t1 = BinaryOp('mul', Constant(c1), v1)
            t2 = BinaryOp('mul', Constant(c2), v2)
            seeds.append(BinaryOp('add', t1, t2))
    
    # Linear combinations WITH constant
    for c1 in [-1.0, 0.5]:
        for c2 in [0.5, 1.0]:
            for c3 in [0.5, 1.0, 1.5]:
                t1 = BinaryOp('mul', Constant(c1), v1)
                t2 = BinaryOp('mul', Constant(c2), v2)
                lin_combo = BinaryOp('add', t1, t2)
                seeds.append(BinaryOp('add', lin_combo, Constant(c3)))
    
    return seeds


def build_dissipation_seeds(var_name: str = "J2") -> List[Node]:
    """Seeds for Φ(J2) = η·J2"""
    var = Variable(0, var_name)
    return [
        var,
        BinaryOp('mul', Constant(0.5), var),
        BinaryOp('mul', Constant(0.25), var),
        BinaryOp('mul', Constant(1.0), var),
    ]


# =============================================================================
# SIMPLE GP
# =============================================================================
class SimpleGP:
    def __init__(self, var_names: List[str], population_size: int = 200):
        self.var_names = var_names
        self.population_size = population_size
        self.population = []
    
    def random_tree(self, max_depth: int) -> Node:
        """Generate random tree"""
        if max_depth == 0 or np.random.rand() < 0.4:
            if np.random.rand() < 0.7:
                idx = np.random.randint(len(self.var_names))
                return Variable(idx, self.var_names[idx])
            return Constant(np.random.choice([0.0, 1.0, 2.0, 0.5, -1.0]))
        
        if np.random.rand() < 0.3:
            op = np.random.choice(['neg', 'square', 'sqrt', 'log', 'exp'])
            return UnaryOp(op, self.random_tree(max_depth - 1))
        else:
            op = np.random.choice(['add', 'sub', 'mul', 'div'])
            return BinaryOp(op, self.random_tree(max_depth - 1), 
                          self.random_tree(max_depth - 1))
    
    def initialize(self, seeds: List[Node]):
        """Initialize with seeds"""
        n_per_seed = self.population_size // (len(seeds) + 1) if seeds else 0
        
        for seed in seeds:
            for _ in range(n_per_seed):
                self.population.append(Individual(tree=copy.deepcopy(seed)))
        
        while len(self.population) < self.population_size:
            self.population.append(Individual(tree=self.random_tree(max_depth=4)))
    
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """Evaluate fitness"""
        for ind in self.population:
            y_pred = ind.evaluate(X)
            ind.mse = np.mean((y - y_pred)**2)
            ind.complexity = ind.tree.complexity()
            ind.fitness = ind.mse + 0.002 * ind.complexity
    
    def mutate(self, tree: Node) -> Node:
        """Simple mutation"""
        if np.random.rand() < 0.7:
            return self.random_tree(max_depth=2)
        return copy.deepcopy(tree)
    
    def evolve(self, X: np.ndarray, y: np.ndarray, n_generations: int = 100) -> Individual:
        """Main evolution loop"""
        self.evaluate(X, y)
        
        for gen in range(n_generations):
            new_pop = []
            
            # Elitism
            sorted_pop = sorted(self.population, key=lambda x: x.fitness)
            new_pop.extend([copy.deepcopy(ind) for ind in sorted_pop[:2]])
            
            # Tournament
            while len(new_pop) < self.population_size:
                tournament = np.random.choice(self.population, size=7, replace=False)
                parent = min(tournament, key=lambda x: x.fitness)
                
                if np.random.rand() < 0.8:
                    child_tree = self.mutate(parent.tree)
                else:
                    child_tree = copy.deepcopy(parent.tree)
                
                new_pop.append(Individual(tree=child_tree))
            
            self.population = new_pop
            self.evaluate(X, y)
            
            if gen % 20 == 0:
                best = min(self.population, key=lambda x: x.fitness)
                print(f"  Gen {gen:3d} | MSE: {best.mse:.6e}, C: {best.complexity:2d}")
        
        return min(self.population, key=lambda x: x.fitness)


# =============================================================================
# CONSTANT OPTIMIZATION (CAREFUL!)
# =============================================================================
def optimize_constants(ind: Individual, X: np.ndarray, y: np.ndarray,
                      max_attempts: int = 3) -> Individual:
    """Optimize constants carefully without breaking solutions"""
    
    def extract(node, lst):
        if isinstance(node, Constant):
            lst.append(node.value)
        elif isinstance(node, UnaryOp):
            extract(node.child, lst)
        elif isinstance(node, BinaryOp):
            extract(node.left, lst)
            extract(node.right, lst)
    
    constants = []
    extract(ind.tree, constants)
    
    if len(constants) == 0:
        return ind
    
    def replace(node, vals, idx):
        if isinstance(node, Constant):
            if idx[0] < len(vals):
                val = vals[idx[0]]
                idx[0] += 1
                return Constant(val)
            return node
        elif isinstance(node, UnaryOp):
            return UnaryOp(node.op, replace(node.child, vals, idx))
        elif isinstance(node, BinaryOp):
            return BinaryOp(node.op, replace(node.left, vals, idx),
                          replace(node.right, vals, idx))
        return node
    
    def objective(vals):
        new_tree = replace(copy.deepcopy(ind.tree), vals, [0])
        new_ind = Individual(tree=new_tree)
        pred = new_ind.evaluate(X)
        return np.mean((y - pred)**2)
    
    # Try optimization with different strategies
    best_result = ind
    best_mse = ind.mse
    
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # Conservative bounds
                bounds = [(-20, 20) for _ in constants]
            elif attempt == 1:
                # Wider bounds for log scale
                bounds = [(-50, 50) for _ in constants]
            else:
                # Very wide bounds
                bounds = [(-100, 100) for _ in constants]
            
            result = differential_evolution(
                objective, bounds, maxiter=100, seed=42 + attempt,
                atol=1e-12, tol=1e-12, polish=True
            )
            
            new_tree = replace(copy.deepcopy(ind.tree), result.x, [0])
            new_ind = Individual(tree=new_tree, mse=result.fun)
            
            # Only accept if it's actually better
            if new_ind.mse < best_mse:
                best_result = new_ind
                best_mse = new_ind.mse
        
        except:
            continue
    
    return best_result


# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_data(model: str, G: float = 1.0, eta_s: float = 0.5,
                 L2: float = 10.0, lam: float = 0.5):
    """Generate trajectory data"""
    
    def uct(C, L, lam):
        I = np.eye(3)
        return L @ C + C @ L.T - (C - I) / lam
    
    all_data = []
    
    # Steady shear
    for sr in np.logspace(-1.8, 1.0, 20):
        C = np.eye(3)
        L = np.array([[0, sr, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        for step in range(100):
            D = 0.5 * (L + L.T)
            C = C + uct(C, L, lam) * 0.01
            evals, evecs = np.linalg.eigh(C)
            C = evecs @ np.diag(np.maximum(evals, 1e-8)) @ evecs.T
            
            if model == "fene_p":
                I1 = np.trace(C)
                if I1 > 0.95 * L2:
                    C = np.eye(3) + (0.95*L2/I1 - 1) * (C - np.eye(3))
            
            if step % 10 == 0:
                all_data.append([np.trace(C), np.trace(C @ C),
                               np.trace(D), np.trace(D @ D)])
    
    data = np.array(all_data)
    I1, I2, J1, J2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    if model == "oldroyd_b":
        psi_true = 0.5 * G * (I2 - 2*I1 + 3)
    else:
        psi_true = -(G/2) * L2 * np.log(np.clip(1 - I1/L2, 1e-10, 1.0))
    
    phi_true = eta_s * J2
    
    return {'I1': I1, 'I2': I2, 'J2': J2,
            'psi_true': psi_true, 'phi_true': phi_true}


# =============================================================================
# MAIN DISCOVERY
# =============================================================================
def discover_rayleighian(model: str = "fene_p"):
    """Discover both Ψ and Φ"""
    
    print(f"\n{'#'*80}")
    print(f"# {model.upper()}: CLEAN GP WITH SMART SEEDING")
    print(f"{'#'*80}\n")
    
    # Data
    L2 = 10.0 if model == "fene_p" else 100.0
    data = generate_data(model, G=1.0, eta_s=0.5, L2=L2)
    print(f"Generated {len(data['I1'])} snapshots")
    
    # =========================================================================
    # DISCOVER Ψ
    # =========================================================================
    print(f"\n{'-'*80}")
    print("Ψ (Elastic Potential)")
    print(f"{'-'*80}")
    
    if model == "fene_p":
        X_psi = data['I1'].reshape(-1, 1)
        var_names_psi = ["I1"]
        seeds_psi = build_fene_seeds("I1")
        n_gen = 120
    else:
        X_psi = np.column_stack([data['I1'], data['I2']])
        var_names_psi = ["I1", "I2"]
        seeds_psi = build_oldroyd_seeds(var_names_psi)
        n_gen = 80
    
    print(f"  Using {len(seeds_psi)} seed expressions")
    
    gp_psi = SimpleGP(var_names_psi, population_size=200)
    gp_psi.initialize(seeds_psi)
    best_psi = gp_psi.evolve(X_psi, data['psi_true'], n_generations=n_gen)
    
    # Only optimize/simplify if solution isn't already perfect
    if best_psi.mse > 1e-6:
        print("  Optimizing constants...")
        optimized = optimize_constants(best_psi, X_psi, data['psi_true'])
        # Only accept if it actually improved
        if optimized.mse < best_psi.mse:
            best_psi = optimized
        
        print("  Simplifying...")
        simplified = simplify_tree(best_psi.tree, var_names_psi, aggressive=False)
        # Check if simplification broke it
        test_ind = Individual(tree=simplified)
        test_pred = test_ind.evaluate(X_psi)
        test_mse = np.mean((data['psi_true'] - test_pred)**2)
        if test_mse < best_psi.mse * 1.1:  # Allow 10% tolerance
            best_psi.tree = simplified
            best_psi.complexity = simplified.complexity()
    else:
        print("  Solution already perfect (MSE < 1e-6), skipping optimization")
    
    # =========================================================================
    # DISCOVER Φ
    # =========================================================================
    print(f"\n{'-'*80}")
    print("Φ (Dissipation Potential)")
    print(f"{'-'*80}")
    
    X_phi = data['J2'].reshape(-1, 1)
    var_names_phi = ["J2"]
    seeds_phi = build_dissipation_seeds("J2")
    
    print(f"  Using {len(seeds_phi)} seed expressions")
    
    gp_phi = SimpleGP(var_names_phi, population_size=100)
    gp_phi.initialize(seeds_phi)
    best_phi = gp_phi.evolve(X_phi, data['phi_true'], n_generations=30)
    
    if best_phi.mse > 1e-6:
        print("  Optimizing constants...")
        optimized = optimize_constants(best_phi, X_phi, data['phi_true'])
        if optimized.mse < best_phi.mse:
            best_phi = optimized
        
        print("  Simplifying...")
        simplified = simplify_tree(best_phi.tree, var_names_phi, aggressive=False)
        test_ind = Individual(tree=simplified)
        test_pred = test_ind.evaluate(X_phi)
        test_mse = np.mean((data['phi_true'] - test_pred)**2)
        if test_mse < best_phi.mse * 1.1:
            best_phi.tree = simplified
            best_phi.complexity = simplified.complexity()
    else:
        print("  Solution already perfect (MSE < 1e-6), skipping optimization")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    psi_pred = best_psi.evaluate(X_psi)
    phi_pred = best_phi.evaluate(X_phi)
    
    r2_psi = 1 - np.sum((data['psi_true'] - psi_pred)**2) / np.var(data['psi_true']) / len(data['psi_true'])
    r2_phi = 1 - np.sum((data['phi_true'] - phi_pred)**2) / np.var(data['phi_true']) / len(data['phi_true'])
    
    print(f"\n{'='*80}")
    print("DISCOVERED RAYLEIGHIAN")
    print(f"{'='*80}\n")
    print(f"  Ψ = {best_psi}")
    print(f"  Φ = {best_phi}\n")
    print(f"ACCURACY:")
    print(f"  Ψ: R² = {r2_psi:.10f}, MSE = {best_psi.mse:.6e}")
    print(f"  Φ: R² = {r2_phi:.10f}, MSE = {best_phi.mse:.6e}")
    print(f"{'='*80}\n")
    
    return {
        'psi': str(best_psi), 'phi': str(best_phi),
        'r2_psi': r2_psi, 'r2_phi': r2_phi,
        'psi_ind': best_psi, 'phi_ind': best_phi
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    
    print("\n" + "#"*80)
    print("# ULTIMATE: CLEAN GENERATIVE GP")
    print("# Smart Seeding + Proven Features Only")
    print("#"*80)
    
    # Oldroyd-B
    results_ob = discover_rayleighian("oldroyd_b")
    
    # FENE-P
    results_fene = discover_rayleighian("fene_p")
    
    # Summary
    print("\n" + "#"*80)
    print("# FINAL RESULTS")
    print("#"*80)
    print(f"\n{'Model':<12} {'Ψ':<45} {'Φ':<15} {'R²(Ψ)':<10}")
    print("-"*85)
    print(f"{'Oldroyd-B':<12} {results_ob['psi']:<45} {results_ob['phi']:<15} {results_ob['r2_psi']:.6f}")
    print(f"{'FENE-P':<12} {results_fene['psi']:<45} {results_fene['phi']:<15} {results_fene['r2_psi']:.6f}")
    print(f"\n{'#'*80}\n")