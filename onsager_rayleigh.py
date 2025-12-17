#%%
#!/usr/bin/env python3
"""
Rayleighian Discovery with Enhanced K-Means Seed Selection (~500 lines)
========================================================================
Optimized for exact recovery with MSE-weighted clustering and adaptive seeding
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import List
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
# COMPREHENSIVE TERM LIBRARY BUILDER
# =============================================================================
def build_term_library(var_names, max_complexity=5):
    """
    Build comprehensive library of low-complexity terms
    Enhanced with more constant variations and physics-motivated forms
    """
    print(f"  Building term library (max complexity {max_complexity})...")
    library = []
    
    # Level 0: More constants
    for c in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, -0.5, -1.0, -2.0, -5.0]:
        library.append((Constant(c), 0))
    
    # Level 1: Variables
    variables = [Variable(i, name) for i, name in enumerate(var_names)]
    for var in variables:
        library.append((var, 1))
    
    if max_complexity >= 2:
        # Level 2: Unary operations on variables
        for var in variables:
            for op in ['neg', 'square', 'sqrt', 'log', 'exp']:
                library.append((UnaryOp(op, copy.deepcopy(var)), 2))
        
        # Level 2: Binary operations (var op const) - MORE constants
        for var in variables:
            for c in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, -0.5, -1.0, -2.0, -5.0]:
                for op in ['add', 'sub', 'mul', 'div']:
                    library.append((BinaryOp(op, copy.deepcopy(var), Constant(c)), 2))
                    library.append((BinaryOp(op, Constant(c), copy.deepcopy(var)), 2))
    
    if max_complexity >= 3:
        # Level 3: Binary operations between variables
        if len(variables) >= 2:
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        for op in ['add', 'sub', 'mul', 'div']:
                            library.append((BinaryOp(op, copy.deepcopy(var1), copy.deepcopy(var2)), 3))
        
        # Level 3: Unary on (var op const)
        for var in variables:
            for c in [0.5, 1.0, 2.0, 3.0]:
                for binop in ['add', 'sub', 'mul', 'div']:
                    base = BinaryOp(binop, copy.deepcopy(var), Constant(c))
                    for unop in ['square', 'sqrt', 'log']:
                        library.append((UnaryOp(unop, copy.deepcopy(base)), 3))
    
    if max_complexity >= 4:
        # Level 4: More complex combinations
        if len(variables) >= 2:
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        # (c1·var1 + c2·var2) combinations
                        for c1 in [-2.0, -1.0, 0.5, 1.0, 2.0]:
                            for c2 in [0.5, 1.0, 2.0]:
                                term1 = BinaryOp('mul', Constant(c1), copy.deepcopy(var1))
                                term2 = BinaryOp('mul', Constant(c2), copy.deepcopy(var2))
                                library.append((BinaryOp('add', term1, term2), 4))
        
        # Level 4: Physics-motivated forms for single variable
        for var in variables:
            # c1 * log(1 - var/c2) with MORE scales
            for c1 in [-10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0]:
                for c2 in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
                    ratio = BinaryOp('div', copy.deepcopy(var), Constant(c2))
                    one_minus = BinaryOp('sub', Constant(1.0), ratio)
                    log_term = UnaryOp('log', one_minus)
                    library.append((BinaryOp('mul', Constant(c1), log_term), 4))
    
    if max_complexity >= 5:
        # Level 5: Even more complex physics forms
        if len(variables) >= 2:
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        # (c1·var1 + c2·var2 + c3) forms
                        for c1 in [-2.0, -1.0, 0.5, 1.0]:
                            for c2 in [0.5, 1.0, 2.0]:
                                for c3 in [1.0, 2.0, 3.0]:
                                    term1 = BinaryOp('mul', Constant(c1), copy.deepcopy(var1))
                                    term2 = BinaryOp('mul', Constant(c2), copy.deepcopy(var2))
                                    sum12 = BinaryOp('add', term1, term2)
                                    library.append((BinaryOp('add', sum12, Constant(c3)), 5))
    
    print(f"  Generated {len(library)} candidate terms")
    return library

# =============================================================================
# ENHANCED K-MEANS SEED SELECTION
# =============================================================================
def select_seeds_kmeans(library, X, y, n_seeds=25, n_samples=100):
    """
    Enhanced k-means with MSE-weighted selection
    """
    print(f"  Selecting {n_seeds} seeds via enhanced k-means clustering...")
    
    # Subsample data for efficiency
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices] if X.ndim == 1 else X[indices, :]
        y_sample = y[indices]
    else:
        X_sample, y_sample = X, y
    
    # Evaluate all terms and build feature matrix
    features = []
    valid_terms = []
    mse_values = []
    
    for tree, complexity in library:
        try:
            ind = Individual(tree=tree)
            pred = ind.evaluate(X_sample)
            
            if not np.all(np.isfinite(pred)):
                continue
            
            # Calculate metrics
            corr = np.corrcoef(pred, y_sample)[0, 1] if np.std(pred) > 1e-10 else 0.0
            mse = np.mean((pred - y_sample)**2)
            
            # Feature vector emphasizing MSE and correlation
            feature_vec = [
                corr if np.isfinite(corr) else 0.0,
                -np.log10(mse + 1e-20),  # Log-scale MSE
                -complexity / 10.0,  # Normalized complexity penalty
                np.std(pred) / (np.std(y_sample) + 1e-10),  # Relative diversity
                np.abs(np.mean(pred) - np.mean(y_sample)) / (np.std(y_sample) + 1e-10),  # Mean error
            ]
            
            features.append(feature_vec)
            valid_terms.append(tree)
            mse_values.append(mse)
            
        except:
            continue
    
    if len(valid_terms) < n_seeds:
        print(f"  Warning: Only {len(valid_terms)} valid terms, using all")
        return valid_terms
    
    # Normalize features
    features = np.array(features)
    mse_values = np.array(mse_values)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means clustering with more clusters than needed
    n_clusters = min(n_seeds * 2, len(valid_terms))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    
    # From each cluster, select the term with LOWEST MSE
    cluster_best = []
    for i in range(n_clusters):
        cluster_mask = kmeans.labels_ == i
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            # Select term with lowest MSE in this cluster
            best_idx = cluster_indices[np.argmin(mse_values[cluster_indices])]
            cluster_best.append((best_idx, mse_values[best_idx]))
    
    # Sort by MSE and take top n_seeds
    cluster_best.sort(key=lambda x: x[1])
    selected_indices = [idx for idx, _ in cluster_best[:n_seeds]]
    selected_trees = [valid_terms[idx] for idx in selected_indices]
    
    # Print selected seeds
    print(f"  Selected seeds (top 15):")
    for i, idx in enumerate(selected_indices[:15]):
        tree = valid_terms[idx]
        ind = Individual(tree=tree)
        pred = ind.evaluate(X_sample)
        mse = np.mean((pred - y_sample)**2)
        print(f"    {i+1:2d}. {ind} (MSE: {mse:.6e}, C: {tree.complexity()})")
    if len(selected_trees) > 15:
        print(f"    ... and {len(selected_trees) - 15} more")
    
    return selected_trees

# =============================================================================
# ENHANCED GP
# =============================================================================
class SimpleGP:
    def __init__(self, var_names, pop_size=200):
        self.var_names, self.pop_size, self.population = var_names, pop_size, []
    
    def random_tree(self, max_depth):
        if max_depth == 0 or np.random.rand() < 0.4:
            return Variable(np.random.randint(len(self.var_names)), 
                          self.var_names[np.random.randint(len(self.var_names))]) \
                   if np.random.rand() < 0.7 else Constant(np.random.choice([0., 1., 2., 0.5, -1., 3., 5.]))
        if np.random.rand() < 0.3:
            return UnaryOp(np.random.choice(['neg', 'square', 'sqrt', 'log', 'exp']),
                          self.random_tree(max_depth - 1))
        return BinaryOp(np.random.choice(['add', 'sub', 'mul', 'div']),
                       self.random_tree(max_depth - 1), self.random_tree(max_depth - 1))
    
    def initialize(self, seeds):
        # Heavily bias toward best seeds
        n_best = min(5, len(seeds))
        copies_per_best = self.pop_size // (n_best * 2)
        
        # Add many copies of top seeds
        for i, seed in enumerate(seeds[:n_best]):
            for _ in range(copies_per_best):
                self.population.append(Individual(tree=copy.deepcopy(seed)))
        
        # Add fewer copies of remaining seeds
        for seed in seeds[n_best:]:
            n_copies = max(1, self.pop_size // (len(seeds) * 3))
            for _ in range(n_copies):
                self.population.append(Individual(tree=copy.deepcopy(seed)))
        
        # Fill rest with random
        while len(self.population) < self.pop_size:
            self.population.append(Individual(tree=self.random_tree(4)))
    
    def evaluate(self, X, y):
        for ind in self.population:
            y_pred = ind.evaluate(X)
            ind.mse, ind.complexity = np.mean((y - y_pred)**2), ind.tree.complexity()
            ind.fitness = ind.mse + 0.001 * ind.complexity  # Reduced complexity penalty
    
    def evolve(self, X, y, n_gen=100, verbose=True):
        self.evaluate(X, y)
        for gen in range(n_gen):
            # Keep top 5 elites
            new_pop = sorted(self.population, key=lambda x: x.fitness)[:5]
            new_pop = [copy.deepcopy(ind) for ind in new_pop]
            
            while len(new_pop) < self.pop_size:
                # Smaller tournament for more exploration
                tournament = np.random.choice(self.population, 5, replace=False)
                parent = min(tournament, key=lambda x: x.fitness)
                
                # Less randomness - more exploitation of good individuals
                if np.random.rand() < 0.7:  # Reduced from 0.8
                    child_tree = copy.deepcopy(parent.tree)
                else:
                    child_tree = self.random_tree(2)
                
                new_pop.append(Individual(tree=child_tree))
            
            self.population = new_pop
            self.evaluate(X, y)
            
            if verbose and gen % 20 == 0:
                best = min(self.population, key=lambda x: x.fitness)
                print(f"  Gen {gen:3d} | MSE: {best.mse:.6e}, C: {best.complexity:2d}, Expr: {str(best)[:60]}")
        
        return min(self.population, key=lambda x: x.fitness)

# =============================================================================
# CONSTANT OPTIMIZATION
# =============================================================================
def optimize_constants(ind, X, y, maxiter=200):
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
        result = differential_evolution(obj, [(-30, 30)]*len(constants), 
                                       maxiter=maxiter, seed=42, atol=1e-14, polish=True)
        return Individual(tree=replace(copy.deepcopy(ind.tree), result.x, [0]), mse=result.fun)
    except: return ind

# =============================================================================
# SIMPLIFICATION
# =============================================================================
def simplify_tree(tree, var_names):
    """Enhanced sympy simplification"""
    try:
        syms = [sp.Symbol(n) for n in var_names]
        expr = tree_to_sympy(tree, syms)
        expr = sp.simplify(expr)
        expr = sp.collect(expr, syms)
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
    print(f"\n{'#'*80}\n# {model.upper()}: ENHANCED K-MEANS SEED SELECTION\n{'#'*80}\n")
    
    L2 = 10.0 if model == "fene_p" else 100.0
    data = generate_data(model, G=1.0, eta_s=0.5, L2=L2)
    print(f"Generated {len(data['I1'])} snapshots\n")
    
    # Discover Ψ
    print(f"{'-'*80}\nΨ (Elastic Potential)\n{'-'*80}")
    if model == "fene_p":
        X_psi = data['I1'].reshape(-1, 1)
        var_names_psi = ["I1"]
        n_gen = 150
        max_complexity = 5
    else:
        X_psi = np.column_stack([data['I1'], data['I2']])
        var_names_psi = ["I1", "I2"]
        n_gen = 100
        max_complexity = 5
    
    # Build library and select seeds
    library = build_term_library(var_names_psi, max_complexity=max_complexity)
    seeds_psi = select_seeds_kmeans(library, X_psi, data['psi_true'], n_seeds=25)
    
    print(f"\n  Training GP with {len(seeds_psi)} k-means selected seeds...")
    gp_psi = SimpleGP(var_names_psi, pop_size=250)
    gp_psi.initialize(seeds_psi)
    best_psi = gp_psi.evolve(X_psi, data['psi_true'], n_gen)
    
    if best_psi.mse > 1e-8:
        print("\n  Optimizing constants (extended)...")
        optimized = optimize_constants(best_psi, X_psi, data['psi_true'], maxiter=200)
        if optimized.mse < best_psi.mse:
            print(f"    Improved MSE: {best_psi.mse:.6e} → {optimized.mse:.6e}")
            best_psi = optimized
        print("  Simplifying...")
        simplified = simplify_tree(best_psi.tree, var_names_psi)
        test_mse = np.mean((data['psi_true'] - Individual(tree=simplified).evaluate(X_psi))**2)
        if test_mse < best_psi.mse * 1.1:
            best_psi.tree = simplified
            best_psi.mse = test_mse
    else:
        print("  Perfect (MSE < 1e-8), skipping optimization")
    
    # Discover Φ
    print(f"\n{'-'*80}\nΦ (Dissipation Potential)\n{'-'*80}")
    X_phi = data['J2'].reshape(-1, 1)
    var_names_phi = ["J2"]
    
    library_phi = build_term_library(var_names_phi, max_complexity=3)
    seeds_phi = select_seeds_kmeans(library_phi, X_phi, data['phi_true'], n_seeds=10)
    
    print(f"\n  Training GP with {len(seeds_phi)} k-means selected seeds...")
    gp_phi = SimpleGP(var_names_phi, pop_size=100)
    gp_phi.initialize(seeds_phi)
    best_phi = gp_phi.evolve(X_phi, data['phi_true'], 30, verbose=False)
    if best_phi.mse > 1e-8:
        best_phi = optimize_constants(best_phi, X_phi, data['phi_true'])
    
    # Results
    psi_pred, phi_pred = best_psi.evaluate(X_psi), best_phi.evaluate(X_phi)
    r2_psi = 1 - np.sum((data['psi_true'] - psi_pred)**2) / np.var(data['psi_true']) / len(data['psi_true'])
    r2_phi = 1 - np.sum((data['phi_true'] - phi_pred)**2) / np.var(data['phi_true']) / len(data['phi_true'])
    
    print(f"\n{'='*80}\nDISCOVERED RAYLEIGHIAN\n{'='*80}\n")
    print(f"  Ψ = {best_psi}\n  Φ = {best_phi}\n")
    print(f"ACCURACY:\n  Ψ: R² = {r2_psi:.10f}, MSE = {best_psi.mse:.6e}")
    print(f"  Φ: R² = {r2_phi:.10f}, MSE = {best_phi.mse:.6e}\n{'='*80}\n")
    
    return {'psi': str(best_psi), 'phi': str(best_phi), 'r2_psi': r2_psi, 'r2_phi': r2_phi, 'mse_psi': best_psi.mse}

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    print("\n" + "#"*80 + "\n# RAYLEIGHIAN DISCOVERY: ENHANCED K-MEANS SEED SELECTION\n" + "#"*80)
    
    results_ob = discover_rayleighian("oldroyd_b")
    results_fene = discover_rayleighian("fene_p")