#%%
#!/usr/bin/env python3
"""
Clean Rheology Discovery
=========================
Discover constitutive equations for Oldroyd-B and FENE-P using PySR
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# TENSOR OPERATIONS & DATA GENERATION
# =============================================================================

def invariants_C(C):
    I1 = np.trace(C, axis1=-2, axis2=-1)
    I2 = np.trace(C @ C, axis1=-2, axis2=-1)
    return I1, I2

def invariants_D(D):
    J1 = np.trace(D, axis1=-2, axis2=-1)
    J2 = np.trace(D @ D, axis1=-2, axis2=-1)
    return J1, J2

def upper_convected_derivative(C, D, lam):
    dim = C.shape[-1]
    I = np.eye(dim)
    return 2 * (D @ C) - (C - I) / lam

def random_symmetric(scale=0.5, dim=3):
    A = np.random.randn(dim, dim) * scale
    return 0.5 * (A + A.T)

def generate_steady_shear_experiment(shear_rate, n_steps, dt, dim, lam, L2, model):
    C = np.eye(dim)
    D = np.zeros((dim, dim))
    D[0, 1] = D[1, 0] = shear_rate / 2.0

    C_traj, D_traj = [], []

    for _ in range(n_steps):
        C_traj.append(C.copy())
        D_traj.append(D.copy())

        dCdt = upper_convected_derivative(C, D, lam)
        C = C + dCdt * dt

        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1e-8)
        C = V @ np.diag(w) @ V.T

        if model == "fene_p":
            I1 = np.trace(C)
            if I1 > 0.95 * L2:
                s = (0.95 * L2) / I1
                C = np.eye(dim) + s * (C - np.eye(dim))

    return np.array(C_traj), np.array(D_traj)

def generate_data(model, seed=42, G=1.0, eta_s=0.5, L2=100.0, lam=0.5):
    np.random.seed(seed)
    dim = 3
    C_arr, D_arr = [], []

    # Dense shear rate coverage
    shear_rates = np.logspace(-1, 0.7, 25)
    for sr in shear_rates:
        C_traj, D_traj = generate_steady_shear_experiment(
            sr, 120, 0.015, dim, lam, L2, model
        )
        C_arr.extend(C_traj)
        D_arr.extend(D_traj)

    if model == "fene_p":
        # High-strain excursions
        for _ in range(15):
            C_traj, D_traj = generate_steady_shear_experiment(
                np.random.uniform(3.0, 6.0), 300, 0.015, dim, lam, L2, model
            )
            C_arr.extend(C_traj)
            D_arr.extend(D_traj)

        # Dense I1 sampling
        I1_low = np.linspace(3.0, 0.5*L2, 150)
        I1_mid = np.linspace(0.5*L2, 0.8*L2, 200)
        I1_high = np.linspace(0.8*L2, 0.95*L2, 250)
        I1_samples = np.concatenate([I1_low, I1_mid, I1_high])
        
        for target_I1 in I1_samples:
            eigenvals = np.array([target_I1/3.0, target_I1/3.0, target_I1/3.0])
            eigenvals += np.random.randn(3) * 0.05
            eigenvals = np.maximum(eigenvals, 0.1)
            eigenvals = eigenvals * (target_I1 / eigenvals.sum())
            
            Q, _ = np.linalg.qr(np.random.randn(dim, dim))
            C = Q @ np.diag(eigenvals) @ Q.T
            D = random_symmetric(scale=0.3)
            
            C_arr.append(C)
            D_arr.append(D)

    I1, I2, J1, J2, psi, phi = [], [], [], [], [], []

    for C, D in zip(C_arr, D_arr):
        i1, i2 = invariants_C(C[None])
        j1, j2 = invariants_D(D[None])
        i1, i2, j1, j2 = map(float, [i1[0], i2[0], j1[0], j2[0]])

        if model == "oldroyd_b":
            psi_val = 0.5 * G * (i2 - 2*i1 + 3)
        else:
            psi_val = -(G/2) * L2 * np.log(max(1e-12, 1 - i1/L2))

        phi_val = eta_s * j2

        I1.append(i1); I2.append(i2)
        J1.append(j1); J2.append(j2)
        psi.append(psi_val); phi.append(phi_val)

    return {
        "I1": np.array(I1), "I2": np.array(I2),
        "J1": np.array(J1), "J2": np.array(J2),
        "psi_true": np.array(psi), "phi_true": np.array(phi),
        "params": dict(G=G, eta_s=eta_s, L2=L2, lam=lam)
    }

# =============================================================================
# DISCOVERY
# =============================================================================

def discover_oldroyd_b(data):
    """Discover Oldroyd-B elastic potential."""
    
    print("\n" + "="*80)
    print("DISCOVERING OLDROYD-B")
    print("="*80 + "\n")
    
    X = np.column_stack([data["I1"], data["I2"]])
    y = data["psi_true"]
    var_names = ["I1", "I2"]
    
    model = PySRRegressor(
        niterations=50,
        populations=25,
        population_size=300,
        maxsize=15,
        maxdepth=8,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        weight_mutate_constant=0.1,
        weight_mutate_operator=0.3,
        weight_add_node=0.2,
        weight_insert_node=0.1,
        weight_delete_node=0.1,
        weight_simplify=0.01,
        weight_randomize=0.01,
        weight_do_nothing=0.0,
        parsimony=0.005,
        model_selection="best",
        verbosity=1,
    )
    
    print(f"Training on {len(y)} data points...")
    model.fit(X, y, variable_names=var_names)
    
    # Evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    expr = sp.simplify(model.sympy())
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Discovered: Ψ = {expr}")
    print(f"R² = {r2:.8f}")
    print(f"Expected:   Ψ = 0.5*I2 - I1 + 1.5")
    
    return model, r2

def discover_fene_p(data):
    """Discover FENE-P elastic potential."""
    
    print("\n" + "="*80)
    print("DISCOVERING FENE-P")
    print("="*80 + "\n")
    
    X = data["I1"][:, None]
    y = data["psi_true"]
    var_names = ["I1"]
    L2 = data["params"]["L2"]
    
    model = PySRRegressor(
        niterations=150,
        populations=80,
        population_size=500,
        maxsize=15,
        maxdepth=7,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "neg"],
        extra_sympy_mappings={"L2": L2},
        weight_mutate_constant=0.1,
        weight_mutate_operator=0.3,
        weight_add_node=0.2,
        weight_insert_node=0.1,
        weight_delete_node=0.1,
        weight_simplify=0.01,
        weight_randomize=0.01,
        weight_do_nothing=0.0,
        parsimony=0.003,
        model_selection="best",
        verbosity=1,
    )
    
    print(f"Training on {len(y)} data points...")
    model.fit(X, y, variable_names=var_names)
    
    # Evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    expr = sp.simplify(model.sympy())
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Discovered: Ψ = {expr}")
    print(f"R² = {r2:.8f}")
    print(f"Expected:   Ψ = -{L2/2}*log(1 - I1/{L2})")
    
    return model, r2

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("RHEOLOGY CONSTITUTIVE DISCOVERY")
    print("="*80)
    
    # Generate data
    print("\nGenerating data...")
    oldroyd_data = generate_data("oldroyd_b", L2=100.0)
    print(f"  Oldroyd-B: {len(oldroyd_data['I1'])} data points")
    
    fene_data = generate_data("fene_p", L2=10.0)
    print(f"  FENE-P:    {len(fene_data['I1'])} data points")
    
    # Discover
    oldroyd_model, r2_oldroyd = discover_oldroyd_b(oldroyd_data)
    fene_model, r2_fene = discover_fene_p(fene_data)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Oldroyd-B:  R² = {r2_oldroyd:.8f}")
    print(f"FENE-P:     R² = {r2_fene:.8f}")
    print("\n✅ Discovery complete!")

if __name__ == "__main__":
    main()