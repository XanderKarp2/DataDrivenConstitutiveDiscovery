# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from sampling import ThermodynamicHMCMC, TensorOp
from SINDy import DynamicSINDy


# ----------------------------
# FENE-P physics (dimensionless)
# ----------------------------
# In dimensionless time s = t/λ:
# A^(1) = dA/ds - L^T A - A L = -( f(tr(A)) A - I )
# f(tr(A)) = (Lmax^2 - 3)/(Lmax^2 - tr(A))
#
# Simple shear: L has only L12 nonzero.


def fene_p_f(A, Lmax, eps=1e-9):
    trA = np.trace(A)
    denom = Lmax**2 - trA
    if denom < eps:
        denom = eps
    return (Lmax**2 - 3.0) / denom


def rhs_A(A, Lgrad, Lmax):
    """dA/ds = L^T A + A L - (fA - I)"""
    f = fene_p_f(A, Lmax)
    I = np.eye(3)
    return (Lgrad.T @ A + A @ Lgrad) - (f * A - I)


def simulate_fenep_simple_shear(Wi, Lmax, dt=2e-3, Lt=4.0, A0=None):
    """
    Generate synthetic dataset for FENE-P in simple shear.
    - Wi sets the magnitude of L12 (since in dimensionless time, L ~ Wi).
    - Lmax is finite extensibility parameter in f(tr(A)).
    Returns:
      A(t): (nt,3,3)
      D(t): (nt,3,3)
      UCD_A(t): (nt,3,3) where UCD_A = A^(1)
      Lgrad: (3,3)
      t: (nt,)
    """
    nt = int(Lt / dt)
    t = np.linspace(0.0, Lt, nt)

    # Velocity gradient tensor with only L12 nonzero
    Lgrad = np.zeros((3, 3))
    Lgrad[0, 1] = Wi  # ONLY Lxy term

    # Rate-of-deformation tensor
    Dconst = 0.5 * (Lgrad + Lgrad.T)
    D = np.zeros((nt, 3, 3))
    D[:] = Dconst

    # Initial conformation tensor
    if A0 is None:
        A = np.zeros((nt, 3, 3))
        A[0] = np.eye(3) + 1e-6 * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    else:
        A = np.zeros((nt, 3, 3))
        A[0] = A0

    # Integrate ODE (Euler; dt is small)
    for k in trange(nt - 1, desc=f"Sim Wi={Wi:g}"):
        A[k + 1] = A[k] + dt * rhs_A(A[k], Lgrad, Lmax)

        # Keep symmetry (numerical hygiene)
        A[k + 1] = 0.5 * (A[k + 1] + A[k + 1].T)

    # Target for learning: upper-convected derivative A^(1) = -(fA - I)
    UCD = np.zeros_like(A)
    I = np.eye(3)
    for k in range(nt):
        f = fene_p_f(A[k], Lmax)
        UCD[k] = -(f * A[k] - I)

    return A, D, UCD, Lgrad, t


# ----------------------------
# Utilities: build & apply learned model
# ----------------------------
def filter_tensor_terms(library):
    out = []
    for term in library:
        if getattr(term, "type", "tensor") == "tensor":
            out.append(term)
    return out


def pretty_print_model(coeffs, library, thresh=1e-10):
    idx = np.where(np.abs(coeffs) > thresh)[0]
    idx = idx[np.argsort(-np.abs(coeffs[idx]))]
    print("\n=== Discovered model for A^(1) ===")
    if len(idx) == 0:
        print("(No nonzero terms found)")
        return
    for j in idx:
        print(f"{coeffs[j]: .6e} * {library[j].name}")


def predict_ucd_from_model(D, A, coeffs, library, sindy_obj):
    Theta = sindy_obj._build_library_matrix(library, D, A)
    Ypred = Theta @ coeffs
    return Ypred.reshape(len(D), 3, 3)


def integrate_learned_model(A0, Dconst, Lgrad, dt, nt, coeffs, library, sindy_obj):
    """
    Given learned A^(1) model:
      A^(1) = RHS_model(D, A)
      dA/ds = L^T A + A L + A^(1)
    Integrate forward.
    """
    A = np.zeros((nt, 3, 3))
    A[0] = A0.copy()

    D = np.zeros((nt, 3, 3))
    D[:] = Dconst

    for k in trange(nt - 1, desc="Sim learned"):
        UCDk = predict_ucd_from_model(D[k:k+1], A[k:k+1], coeffs, library, sindy_obj)[0]
        dA = (Lgrad.T @ A[k] + A[k] @ Lgrad) + UCDk
        A[k + 1] = A[k] + dt * dA
        A[k + 1] = 0.5 * (A[k + 1] + A[k + 1].T)

    return A


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # ---- FENE-P parameters ----
    # Lmax is "finite extensibility parameter" in the paper's f(tr(A)) :contentReference[oaicite:2]{index=2}
    Lmax = 20.0

    # We'll train on multiple Wi values (still simple shear, only L12 != 0)
    Wi_train = [0.5, 1.0, 2.0, 4.0]

    dt = 2e-3
    Lt = 4.0

    # ---- Generate training dataset by concatenating trajectories ----
    A_all, D_all, UCD_all = [], [], []

    for Wi in Wi_train:
        A, D, UCD, Lgrad, t = simulate_fenep_simple_shear(Wi, Lmax, dt=dt, Lt=Lt)
        # optional subsample to keep runtime light
        stride = 2
        A_all.append(A[::stride])
        D_all.append(D[::stride])
        UCD_all.append(UCD[::stride])

    A_all = np.concatenate(A_all, axis=0)
    D_all = np.concatenate(D_all, axis=0)
    UCD_all = np.concatenate(UCD_all, axis=0)

    print(f"\nTraining samples: {len(A_all)}")

    # ---- Build candidate symbolic library from sampling.py ----
    sampler = ThermodynamicHMCMC(temperature=0.7, max_complexity=4)

    # Add identity tensor so the model can represent "+ I" type terms
    sampler.tensor_library.append(
        TensorOp("I", lambda D, tau, p: np.eye(3), complexity=1)
    )

    library_raw = sampler.generate_library(N=35)
    library = filter_tensor_terms(library_raw)

    # ---- Fit SINDy on the constitutive law: A^(1) = RHS(D, A) ----
    sindy = DynamicSINDy(threshold=5e-3, lambda_reg=1e-2, max_iter=10)
    coeffs, best_library, best_loss = sindy.fit(D_all, A_all, UCD_all, library)

    pretty_print_model(coeffs, best_library, thresh=1e-8)
    print(f"\nBest loss: {best_loss:.6e}")

    # ---- Compare true vs predicted A^(1) on training set ----
    UCD_pred = predict_ucd_from_model(D_all, A_all, coeffs, best_library, sindy)
    err = np.sqrt(np.mean((UCD_all - UCD_pred) ** 2))
    print(f"RMS error on A^(1): {err:.6e}")

    # ---- Holdout test on an unseen Wi ----
    Wi_test = 3.0
    A_true, D_true, UCD_true, Lgrad, t = simulate_fenep_simple_shear(Wi_test, Lmax, dt=dt, Lt=Lt)
    Dconst = D_true[0]

    A_learned = integrate_learned_model(
        A0=A_true[0],
        Dconst=Dconst,
        Lgrad=Lgrad,
        dt=dt,
        nt=len(t),
        coeffs=coeffs,
        library=best_library,
        sindy_obj=sindy,
    )

    # ---- Plot: compare key components ----
    def plot_comp(ax, arr_true, arr_pred, i, j, title):
        ax.plot(t, arr_true[:, i, j], label="true")
        ax.plot(t, arr_pred[:, i, j], "--", label="learned")
        ax.set_title(title)
        ax.grid(True)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    plot_comp(axs[0, 0], A_true, A_learned, 0, 0, "A11")
    plot_comp(axs[0, 1], A_true, A_learned, 0, 1, "A12 (shear component)")
    plot_comp(axs[1, 0], A_true, A_learned, 1, 1, "A22")
    plot_comp(axs[1, 1], A_true, A_learned, 2, 2, "A33")
    axs[0, 0].legend()
    plt.tight_layout()
    plt.show()
