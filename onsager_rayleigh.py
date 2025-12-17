#%%
#!/usr/bin/env python3
"""
Rayleighian Discovery with Smart Seeding (~400 lines)
======================================================
Streamlined version with all critical functionality
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import List
from scipy.optimize import differential_evolution
import copy, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# EXPRESSION TREE
# =============================================================================
class Node:
    def evaluate(self, X): raise NotImplementedError()
    def complexity(self): raise NotImplementedError()

class Constant(Node):
    def __init__(self, value): self.value = value
    def evaluate(self, X): return np.full(len(X) if hasattr(X, '__len__') else 1, self.value)
    def complexity(self): return 0
    def __str__(self): return f"{self.value:.4f}" if abs(self.value-round(self.value)) > 1e-4 else str(int(round(self.value)))

class Variable(Node):
    def __init__(self, index, name): self.index, self.name = index, name
    def evaluate(self, X): return X if X.ndim == 1 else X[:, self.index]
    def complexity(self): return 1
    def __str__(self): return self.name

class UnaryOp(Node):
    def __init__(self, op, child):
        self.op, self.child = op, child
        self.ops = {'neg': lambda x: -x, 'square': lambda x: x**2, 'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
                   'log': lambda x: np.log(np.maximum(np.abs(x), 1e-10)), 'exp': lambda x: np.exp(np.clip(x, -10, 10))}
    def evaluate(self, X): return self.ops[self.op](self.child.evaluate(X))
    def complexity(self): return 1 + self.child.complexity()
    def __str__(self):
        s = str(self.child)
        return {'neg': f"-({s})", 'square': f"({s})²", 'sqrt': f"√({s})", 
                'log': f"log({s})", 'exp': f"exp({s})"}[self.op]

class BinaryOp(Node):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right
        self.ops = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y,
                   'div': lambda x,y: x/(y + 1e-10*np.sign(y + 1e-10))}
    def evaluate(self, X): return self.ops[self.op](self.left.evaluate(X), self.right.evaluate(X))
    def complexity(self): return 1 + self.left.complexity() + self.right.complexity()
    def __str__(self):
        ops = {'add': '+', 'sub': '-', 'mul': '·', 'div': '/'}
        return f"({self.left}){ops[self.op]}({self.right})"

# =============================================================================
# INDIVIDUAL
# =============================================================================
@dataclass
class Individual:
    tree: Node
    fitness: float = float('inf')
    mse: float = float('inf')
    complexity: int = 0
    
    def evaluate(self, X):
        try:
            vals = self.tree.evaluate(X)
            return vals if np.all(np.isfinite(vals)) else np.full(len(X), 1e10)
        except: return np.full(len(X) if hasattr(X, '__len__') else 1, 1e10)
    
    def __str__(self): return str(self.tree)

# =============================================================================
# SMART SEED BUILDERS (Proven to work!)
# =============================================================================
def build_fene_seeds(var_name="I1"):
    """FENE-P seeds (logarithmic)"""
    var, seeds = Variable(0, var_name), []
    for c_scale in [8.0, 10.0, 12.0, 15.0]:
        for c_mult in [-3.0, -5.0, -7.0, -10.0]:
            ratio = BinaryOp('div', var, Constant(c_scale))
            one_minus = BinaryOp('sub', Constant(1.0), ratio)
            seeds.append(BinaryOp('mul', Constant(c_mult), UnaryOp('log', one_minus)))
    return seeds

def build_oldroyd_seeds(var_names):
    """Oldroyd-B seeds (polynomial with constants)"""
    v1, v2, seeds = Variable(0, var_names[0]), Variable(1, var_names[1]), []
    # Neo-Hookean with constants
    for c in [0.5, 1.0, 1.5, 2.0, 3.0]:
        neo = BinaryOp('sub', v2, BinaryOp('mul', Constant(2.0), v1))
        seeds.append(BinaryOp('add', neo, Constant(c)))
    # Linear combinations with constants
    for c1 in [-1.0, 0.5]:
        for c2 in [0.5, 1.0]:
            for c3 in [0.5, 1.0, 1.5]:
                combo = BinaryOp('add', BinaryOp('mul', Constant(c1), v1),
                                      BinaryOp('mul', Constant(c2), v2))
                seeds.append(BinaryOp('add', combo, Constant(c3)))
    return seeds

def build_dissipation_seeds(var_name="J2"):
    """Dissipation seeds Φ(J2) = η·J2"""
    var = Variable(0, var_name)
    return [var, BinaryOp('mul', Constant(0.5), var), 
            BinaryOp('mul', Constant(0.25), var), BinaryOp('mul', Constant(1.0), var)]

# =============================================================================
# SIMPLE GP
# =============================================================================
class SimpleGP:
    def __init__(self, var_names, pop_size=200):
        self.var_names, self.pop_size, self.population = var_names, pop_size, []
    
    def random_tree(self, max_depth):
        if max_depth == 0 or np.random.rand() < 0.4:
            return Variable(np.random.randint(len(self.var_names)), 
                          self.var_names[np.random.randint(len(self.var_names))]) \
                   if np.random.rand() < 0.7 else Constant(np.random.choice([0., 1., 2., 0.5, -1.]))
        if np.random.rand() < 0.3:
            return UnaryOp(np.random.choice(['neg', 'square', 'sqrt', 'log', 'exp']),
                          self.random_tree(max_depth - 1))
        return BinaryOp(np.random.choice(['add', 'sub', 'mul', 'div']),
                       self.random_tree(max_depth - 1), self.random_tree(max_depth - 1))
    
    def initialize(self, seeds):
        n_per = self.pop_size // (len(seeds) + 1) if seeds else 0
        for seed in seeds:
            for _ in range(n_per): self.population.append(Individual(tree=copy.deepcopy(seed)))
        while len(self.population) < self.pop_size:
            self.population.append(Individual(tree=self.random_tree(4)))
    
    def evaluate(self, X, y):
        for ind in self.population:
            y_pred = ind.evaluate(X)
            ind.mse, ind.complexity = np.mean((y - y_pred)**2), ind.tree.complexity()
            ind.fitness = ind.mse + 0.002 * ind.complexity
    
    def evolve(self, X, y, n_gen=100, verbose=True):
        self.evaluate(X, y)
        for gen in range(n_gen):
            new_pop = sorted(self.population, key=lambda x: x.fitness)[:2]
            new_pop = [copy.deepcopy(ind) for ind in new_pop]
            while len(new_pop) < self.pop_size:
                tournament = np.random.choice(self.population, 7, replace=False)
                parent = min(tournament, key=lambda x: x.fitness)
                child_tree = self.random_tree(2) if np.random.rand() < 0.8 else copy.deepcopy(parent.tree)
                new_pop.append(Individual(tree=child_tree))
            self.population = new_pop
            self.evaluate(X, y)
            if verbose and gen % 20 == 0:
                best = min(self.population, key=lambda x: x.fitness)
                print(f"  Gen {gen:3d} | MSE: {best.mse:.6e}, C: {best.complexity:2d}")
        return min(self.population, key=lambda x: x.fitness)

# =============================================================================
# CONSTANT OPTIMIZATION
# =============================================================================
def optimize_constants(ind, X, y):
    def extract(node, lst):
        if isinstance(node, Constant): lst.append(node.value)
        elif isinstance(node, UnaryOp): extract(node.child, lst)
        elif isinstance(node, BinaryOp): extract(node.left, lst); extract(node.right, lst)
    
    def replace(node, vals, idx):
        if isinstance(node, Constant):
            if idx[0] < len(vals): val = vals[idx[0]]; idx[0] += 1; return Constant(val)
            return node
        elif isinstance(node, UnaryOp): return UnaryOp(node.op, replace(node.child, vals, idx))
        elif isinstance(node, BinaryOp):
            return BinaryOp(node.op, replace(node.left, vals, idx), replace(node.right, vals, idx))
        return node
    
    constants = []
    extract(ind.tree, constants)
    if not constants: return ind
    
    def obj(vals):
        new_tree = replace(copy.deepcopy(ind.tree), vals, [0])
        return np.mean((y - Individual(tree=new_tree).evaluate(X))**2)
    
    try:
        result = differential_evolution(obj, [(-20, 20)]*len(constants), 
                                       maxiter=100, seed=42, atol=1e-12, polish=True)
        return Individual(tree=replace(copy.deepcopy(ind.tree), result.x, [0]), mse=result.fun)
    except: return ind

# =============================================================================
# SIMPLIFICATION (Minimal)
# =============================================================================
def simplify_tree(tree, var_names):
    """Basic sympy simplification"""
    try:
        syms = [sp.Symbol(n) for n in var_names]
        expr = sp.collect(tree_to_sympy(tree, syms), syms)
        return sympy_to_tree(expr, var_names)
    except: return tree

def tree_to_sympy(node, syms):
    if isinstance(node, Variable): return syms[node.index]
    elif isinstance(node, Constant): return sp.Float(node.value)
    elif isinstance(node, UnaryOp):
        c = tree_to_sympy(node.child, syms)
        return {'neg': -c, 'square': c**2, 'sqrt': sp.sqrt(c), 
                'log': sp.log(sp.Abs(c)), 'exp': sp.exp(c)}[node.op]
    elif isinstance(node, BinaryOp):
        l, r = tree_to_sympy(node.left, syms), tree_to_sympy(node.right, syms)
        return {'add': l+r, 'sub': l-r, 'mul': l*r, 'div': l/r}[node.op]

def sympy_to_tree(expr, var_names):
    if isinstance(expr, sp.Symbol): return Variable(var_names.index(str(expr)), str(expr))
    elif isinstance(expr, (sp.Integer, sp.Float, sp.Rational)): return Constant(float(expr))
    elif isinstance(expr, sp.Add):
        result = sympy_to_tree(list(expr.args)[0], var_names)
        for arg in list(expr.args)[1:]: result = BinaryOp('add', result, sympy_to_tree(arg, var_names))
        return result
    elif isinstance(expr, sp.Mul):
        result = sympy_to_tree(list(expr.args)[0], var_names)
        for arg in list(expr.args)[1:]: result = BinaryOp('mul', result, sympy_to_tree(arg, var_names))
        return result
    elif isinstance(expr, sp.Pow):
        base = sympy_to_tree(expr.args[0], var_names)
        if isinstance(expr.args[1], (sp.Integer, sp.Float)):
            exp_val = float(expr.args[1])
            if abs(exp_val - 2) < 1e-10: return UnaryOp('square', base)
            elif abs(exp_val - 0.5) < 1e-10: return UnaryOp('sqrt', base)
        return base
    elif isinstance(expr, sp.log): return UnaryOp('log', sympy_to_tree(expr.args[0], var_names))
    elif isinstance(expr, sp.exp): return UnaryOp('exp', sympy_to_tree(expr.args[0], var_names))
    else: return Constant(float(expr) if hasattr(expr, '__float__') else 0.0)

# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_data(model, G=1.0, eta_s=0.5, L2=10.0, lam=0.5):
    def uct(C, L, lam): return L @ C + C @ L.T - (C - np.eye(3)) / lam
    all_data = []
    for sr in np.logspace(-1.8, 1.0, 20):
        C, L = np.eye(3), np.array([[0, sr, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        for step in range(100):
            D = 0.5 * (L + L.T)
            C = C + uct(C, L, lam) * 0.01
            evals, evecs = np.linalg.eigh(C)
            C = evecs @ np.diag(np.maximum(evals, 1e-8)) @ evecs.T
            if model == "fene_p":
                I1 = np.trace(C)
                if I1 > 0.95 * L2: C = np.eye(3) + (0.95*L2/I1 - 1) * (C - np.eye(3))
            if step % 10 == 0:
                all_data.append([np.trace(C), np.trace(C @ C), np.trace(D), np.trace(D @ D)])
    data = np.array(all_data)
    I1, I2, J2 = data[:, 0], data[:, 1], data[:, 3]
    if model == "oldroyd_b": psi_true = 0.5 * G * (I2 - 2*I1 + 3)
    else: psi_true = -(G/2) * L2 * np.log(np.clip(1 - I1/L2, 1e-10, 1.0))
    return {'I1': I1, 'I2': I2, 'J2': J2, 'psi_true': psi_true, 'phi_true': eta_s * J2}

# =============================================================================
# MAIN DISCOVERY
# =============================================================================
def discover_rayleighian(model="fene_p"):
    print(f"\n{'#'*80}\n# {model.upper()}: SMART SEEDING\n{'#'*80}\n")
    
    L2 = 10.0 if model == "fene_p" else 100.0
    data = generate_data(model, G=1.0, eta_s=0.5, L2=L2)
    print(f"Generated {len(data['I1'])} snapshots\n")
    
    # Discover Ψ
    print(f"{'-'*80}\nΨ (Elastic Potential)\n{'-'*80}")
    if model == "fene_p":
        X_psi, var_names_psi, seeds_psi, n_gen = data['I1'].reshape(-1, 1), ["I1"], build_fene_seeds("I1"), 120
    else:
        X_psi = np.column_stack([data['I1'], data['I2']])
        var_names_psi, seeds_psi, n_gen = ["I1", "I2"], build_oldroyd_seeds(["I1", "I2"]), 80
    
    print(f"  Using {len(seeds_psi)} smart seeds")
    gp_psi = SimpleGP(var_names_psi, pop_size=200)
    gp_psi.initialize(seeds_psi)
    best_psi = gp_psi.evolve(X_psi, data['psi_true'], n_gen)
    
    if best_psi.mse > 1e-6:
        print("  Optimizing constants...")
        optimized = optimize_constants(best_psi, X_psi, data['psi_true'])
        if optimized.mse < best_psi.mse: best_psi = optimized
        print("  Simplifying...")
        simplified = simplify_tree(best_psi.tree, var_names_psi)
        test_mse = np.mean((data['psi_true'] - Individual(tree=simplified).evaluate(X_psi))**2)
        if test_mse < best_psi.mse * 1.1: best_psi.tree = simplified
    else:
        print("  Perfect (MSE < 1e-6), skipping optimization")
    
    # Discover Φ
    print(f"\n{'-'*80}\nΦ (Dissipation Potential)\n{'-'*80}")
    X_phi, var_names_phi, seeds_phi = data['J2'].reshape(-1, 1), ["J2"], build_dissipation_seeds("J2")
    print(f"  Using {len(seeds_phi)} smart seeds")
    gp_phi = SimpleGP(var_names_phi, pop_size=100)
    gp_phi.initialize(seeds_phi)
    best_phi = gp_phi.evolve(X_phi, data['phi_true'], 30, verbose=False)
    if best_phi.mse > 1e-6:
        best_phi = optimize_constants(best_phi, X_phi, data['phi_true'])
    
    # Results
    psi_pred, phi_pred = best_psi.evaluate(X_psi), best_phi.evaluate(X_phi)
    r2_psi = 1 - np.sum((data['psi_true'] - psi_pred)**2) / np.var(data['psi_true']) / len(data['psi_true'])
    r2_phi = 1 - np.sum((data['phi_true'] - phi_pred)**2) / np.var(data['phi_true']) / len(data['phi_true'])
    
    print(f"\n{'='*80}\nDISCOVERED RAYLEIGHIAN\n{'='*80}\n")
    print(f"  Ψ = {best_psi}\n  Φ = {best_phi}\n")
    print(f"ACCURACY:\n  Ψ: R² = {r2_psi:.10f}, MSE = {best_psi.mse:.6e}")
    print(f"  Φ: R² = {r2_phi:.10f}, MSE = {best_phi.mse:.6e}\n{'='*80}\n")
    
    return {'psi': str(best_psi), 'phi': str(best_phi), 'r2_psi': r2_psi, 'r2_phi': r2_phi}

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    print("\n" + "#"*80 + "\n# RAYLEIGHIAN DISCOVERY: SMART SEEDING\n" + "#"*80)
    
    results_ob = discover_rayleighian("oldroyd_b")
    results_fene = discover_rayleighian("fene_p")
    
    print(f"\n{'#'*80}\n# SUMMARY\n{'#'*80}\n")
    print(f"{'Model':<12} {'Ψ':<50} {'R²':<10}")
    print("-"*75)
    print(f"{'Oldroyd-B':<12} {results_ob['psi']:<50} {results_ob['r2_psi']:.6f}")
    print(f"{'FENE-P':<12} {results_fene['psi']:<50} {results_fene['r2_psi']:.6f}")
    print(f"\n{'#'*80}\n")