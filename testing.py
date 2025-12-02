#%%
import numpy as np
import matplotlib.pyplot as plt

from sampling import ThermodynamicHMCMC
from SINDy import DynamicSINDy

def simulate_model(model, n=1000, gamma_dot=2.0):
    """Generate training data with proper stress buildup"""
    D_shear = np.array([[0, gamma_dot/2, 0],
                        [gamma_dot/2, 0, 0],
                        [0, 0, 0]])
    
    W_shear = np.array([[0, gamma_dot/2, 0],
                        [-gamma_dot/2, 0, 0],
                        [0, 0, 0]])
    
    D_arr = np.zeros((n, 3, 3))
    tau_arr = np.zeros((n, 3, 3))
    UCD_arr = np.zeros((n, 3, 3))
    
    # Start from ZERO stress - let it build up naturally
    tau = np.array([[0.01, 0.01, 0],
                    [0.01, 0.01, 0],
                    [0, 0, 0.01]])
    
    dt = 0.01  # Larger timestep for faster buildup
    lambda_ = 1.0  # Larger relaxation time
    
    if model == "Oldroyd-B":
        eta_s = 1.0
        for t in range(n):
            D_arr[t] = D_shear
            tau_arr[t] = tau
            UCD = (2*eta_s/lambda_) * D_shear - (1/lambda_) * tau
            UCD_arr[t] = UCD
            
            if t < n - 1:
                dtau = UCD + (D_shear @ tau + tau @ D_shear) + (W_shear @ tau - tau @ W_shear)
                tau = tau + dtau * dt
                tau = (tau + tau.T) / 2
        
        target = f"UCD(τ) = {2*eta_s/lambda_:.1f}·D - {1/lambda_:.1f}·τ"
    
    elif model == "Giesekus":
        alpha = 0.05
        eta_p = 1.0
        
        for t in range(n):
            D_arr[t] = D_shear
            tau_arr[t] = tau
            UCD = -(1/lambda_) * tau - (alpha/(lambda_*eta_p)) * (tau @ tau)
            UCD_arr[t] = UCD
            
            if t < n - 1:
                dtau = UCD + (D_shear @ tau + tau @ D_shear) + (W_shear @ tau - tau @ W_shear)
                tau = tau + dtau * dt
                tau = (tau + tau.T) / 2
        
        target = f"UCD(τ) = -{1/lambda_:.1f}·τ - {alpha/(lambda_*eta_p):.2f}·(τ@τ)"
    
    elif model == "FENE-P":
        L_sq = 100.0
        eta_p = 2.0  # Polymer viscosity
        
        for t in range(n):
            D_arr[t] = D_shear
            tau_arr[t] = tau
            
            tr_tau = np.trace(tau)
            f = L_sq / max(L_sq - tr_tau, 1.0)
            
            # Full FENE-P constitutive equation
            UCD = (2*eta_p/lambda_) * D_shear - (1/lambda_) * f * tau
            UCD_arr[t] = UCD
            
            if t < n - 1:
                dtau = UCD + (D_shear @ tau + tau @ D_shear) + (W_shear @ tau - tau @ W_shear)
                tau = tau + dtau * dt
                tau = (tau + tau.T) / 2
        
        target = f"UCD(τ) = {2*eta_p/lambda_:.1f}·D - (1/λ)·f·τ where f = {L_sq:.0f}/({L_sq:.0f}-tr(τ))"
    
    return D_arr, tau_arr, UCD_arr, target


def discover_equation(D, tau, UCD, temperature=1.5, n_terms=25):
    
    # Generate library
    hmcmc = ThermodynamicHMCMC(temperature=temperature, max_complexity=4)
    library = hmcmc.generate_library(N=n_terms)
    
    # Run dynamic SINDy
    sindy = DynamicSINDy(threshold=0.05, lambda_reg=0.01, max_iter=20)
    coeffs, final_library, loss = sindy.fit(D, tau, UCD, library)
    
    # Format equation
    active_idx = np.where(np.abs(coeffs) > sindy.threshold / 2)[0]
    sorted_idx = active_idx[np.argsort(-np.abs(coeffs[active_idx]))]
    
    eq_parts = []
    for idx in sorted_idx:
        c = coeffs[idx]
        term = final_library[idx]
        
        # Format with learned parameters
        if len(term.params) > 0:
            term_str = term.name
            for i, val in enumerate(term.params):
                term_str = term_str.replace(f'c{i}', f'{val:.2f}')
            eq_parts.append(f"{c:+.4f}·{term_str}")
        else:
            eq_parts.append(f"{c:+.4f}·{term.name}")
    
    equation = "UCD(τ) = " + " ".join(eq_parts).replace("+ -", "- ")
    if not eq_parts:
        equation = "UCD(τ) = 0"
    
    # Compute errors
    Theta = sindy._build_library_matrix(final_library, D, tau)
    Y = UCD.reshape(-1)
    Y_pred = Theta @ coeffs
    
    errors = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            idx_comp = np.arange(len(D)) * 9 + i*3 + j
            errors[i, j] = np.sqrt(np.mean((Y[idx_comp] - Y_pred[idx_comp])**2))
    
    UCD_pred = Y_pred.reshape(UCD.shape)
    
    return equation, errors, UCD_pred, final_library, coeffs


def plot_results(model, D, tau, UCD_true, UCD_pred, target, discovered, errors):
    """Clean visualization"""
    
    fig = plt.figure(figsize=(16, 8))
    t = np.arange(len(tau)) * 0.005
    
    # Stress evolution
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(t, tau[:, 0, 0], label='τ₁₁', linewidth=2, color='#1f77b4')
    ax1.plot(t, tau[:, 1, 1], label='τ₂₂', linewidth=2, color='#ff7f0e')
    ax1.plot(t, tau[:, 0, 1], label='τ₁₂', linewidth=2, color='#2ca02c')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Stress', fontsize=11)
    ax1.set_title(f'{model}: Stress Evolution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Normal stress difference
    ax2 = plt.subplot(2, 4, 2)
    N1 = tau[:, 0, 0] - tau[:, 1, 1]
    ax2.plot(t, N1, linewidth=2.5, color='#d62728')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('N₁ = τ₁₁ - τ₂₂', fontsize=11)
    ax2.set_title('First Normal Stress Diff', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Steady state check
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(t, tau[:, 0, 1], linewidth=2.5, color='#9467bd')
    final_val = np.mean(tau[-100:, 0, 1])
    ax3.axhline(final_val, color='r', linestyle='--', alpha=0.6, linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('τ₁₂', fontsize=11)
    ax3.set_title(f'Steady: {final_val:.4f}', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Error heatmap
    ax4 = plt.subplot(2, 4, 4)
    im = ax4.imshow(np.log10(errors + 1e-15), cmap='RdYlGn_r', vmin=-12, vmax=-2)
    ax4.set_title('log₁₀(RMSE)', fontsize=12, fontweight='bold')
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    for i in range(3):
        for j in range(3):
            color = 'white' if np.log10(errors[i,j] + 1e-15) < -6 else 'black'
            ax4.text(j, i, f'{errors[i,j]:.1e}', ha="center", va="center",
                    fontsize=9, color=color, weight='bold')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # Scatter plots for key components
    for idx, (i, j, label) in enumerate([(0,0,'τ₁₁'), (1,1,'τ₂₂'), (0,1,'τ₁₂'), (2,2,'τ₃₃')]):
        ax = plt.subplot(2, 4, 5 + idx)
        ax.scatter(UCD_true[:, i, j], UCD_pred[:, i, j], alpha=0.4, s=8, color='#1f77b4')
        lim = max(abs(UCD_true[:, i, j]).max(), abs(UCD_pred[:, i, j]).max()) * 1.1
        if lim > 1e-10:
            ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, linewidth=1.5)
            ax.axis('equal')
        ax.set_xlabel(f'True UCD({label})', fontsize=10)
        ax.set_ylabel(f'Pred UCD({label})', fontsize=10)
        ax.set_title(f'RMSE={errors[i,j]:.2e}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Equation display
    fig.text(0.5, 0.98, f'{model} EQUATION DISCOVERY', 
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.95, f'Target: {target}',
             ha='center', va='top', fontsize=11, color='green', family='monospace')
    fig.text(0.5, 0.92, f'Discovered: {discovered}',
             ha='center', va='top', fontsize=11, color='blue', family='monospace')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    
    print("\n" + "="*80)
    print("THERMODYNAMIC HMCMC-SINDY WITH LEARNABLE CONSTANTS")
    print("="*80)
    
    models = ["Oldroyd-B", "Giesekus", "FENE-P"]
    
    # --- Step 1: Generate ONE expansive library ---
    print("\nGenerating HMCMC library for all models...")
    hmcmc = ThermodynamicHMCMC(temperature=1.5, max_complexity=4)
    library = hmcmc.generate_library(N=50)  # larger pool to cover all models
    
    # --- Step 2: Loop through models ---
    for model in models:
        print("\n" + "="*80)
        print(f"{model.upper()}")
        print("="*80)
        
        # Generate training data
        D, tau, UCD, target = simulate_model(model, n=1000)
        
        # Steady state check (optional)
        tr_tau_final = np.mean([np.trace(tau[i]) for i in range(-100, 0)])
        
        # Discover equation using prebuilt library
        sindy = DynamicSINDy(threshold=0.025, lambda_reg=0.01, max_iter=20)
        coeffs, final_library, loss = sindy.fit(D, tau, UCD, library)
        
        # Format discovered equation
        active_idx = np.where(np.abs(coeffs) > sindy.threshold / 2)[0]
        sorted_idx = active_idx[np.argsort(-np.abs(coeffs[active_idx]))]
        
        eq_parts = []
        for idx in sorted_idx:
            c = coeffs[idx]
            term = final_library[idx]
            if len(term.params) > 0:
                term_str = term.name
                for i, val in enumerate(term.params):
                    term_str = term_str.replace(f'c{i}', f'{val:.2f}')
                eq_parts.append(f"{c:+.4f}·{term_str}")
            else:
                eq_parts.append(f"{c:+.4f}·{term.name}")
        
        equation = "UCD(τ) = " + " ".join(eq_parts).replace("+ -", "- ")
        if not eq_parts:
            equation = "UCD(τ) = 0"
        
        # Compute errors
        Theta = sindy._build_library_matrix(final_library, D, tau)
        Y = UCD.reshape(-1)
        Y_pred = Theta @ coeffs
        errors = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                idx_comp = np.arange(len(D)) * 9 + i*3 + j
                errors[i, j] = np.sqrt(np.mean((Y[idx_comp] - Y_pred[idx_comp])**2))
        UCD_pred = Y_pred.reshape(UCD.shape)
        
        # Print results
        print(f"\nDiscovered:\n  {equation}")
        active_idx = np.where(np.abs(coeffs) > 0.025)[0]
        print(f"\nActive terms with learned constants:")
        for idx in active_idx:
            term = final_library[idx]
            if len(term.params) > 0:
                print(f"  {term.name}: params={term.params}")
        
        rmse = np.sqrt(np.mean(errors**2))
        print(f"\nAccuracy:")
        print(f"  Overall RMSE: {rmse:.6e}")
        print(f"  Component RMSEs: [0,0]={errors[0,0]:.2e}, [1,1]={errors[1,1]:.2e}, [0,1]={errors[0,1]:.2e}")
        
        # Plot
        plot_results(model, D, tau, UCD, UCD_pred, target, equation, errors)

# %%
