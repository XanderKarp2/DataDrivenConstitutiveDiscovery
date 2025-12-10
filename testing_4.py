import os
import numpy as np
from copy import deepcopy
from tqdm import trange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sampling import ThermodynamicHMCMC, TensorOp
from SINDy import DynamicSINDy
from Bindy import BINDyRJ   


SEED = 0
np.random.seed(SEED)

DT = 1e-3
LT = 6.0

ETA = 0.01
LM_TRUTH = 5.0

WI_LIST = [0.5, 1.0, 2.0, 4.0]
GDOT_LIST = [0.5, 1.0, 2.0]
TAU0_LIST = [
    np.array([[0.2, 0.1, 0.0],
              [0.1, 0.3, 0.0],
              [0.0, 0.0, 0.1]], dtype=float),
    np.array([[0.05, 0.02, 0.0],
              [0.02, 0.08, 0.0],
              [0.0, 0.0, 0.03]], dtype=float),
]

OUTDIR = "results_hmcmc_v2"
os.makedirs(OUTDIR, exist_ok=True)

# >>> KEY FIX 1: lower threshold so 0.007-ish scaled trace terms survive
THRESHOLD = 0.001
LAMBDA_REG = 1e-4

HMCMC_TEMPERATURE = 1.0
HMCMC_MAX_COMPLEXITY = 3
HMCMC_N_TERMS = 500

PRUNE_SUBSAMPLE_SAMPLES = 4000
# >>> KEY FIX 2: prune less aggressively so we keep more candidates
COL_DUP_COS_THR = 0.99995
CAP_LIBRARY = 120

FIT_SAMPLE_CAP = 20000


def make_simple_shear_L(gamma_dot: float) -> np.ndarray:
    return np.array([[0.0, gamma_dot, 0.0],
                     [0.0, 0.0,      0.0],
                     [0.0, 0.0,      0.0]], dtype=float)


def sym(A):
    return 0.5 * (A + A.T)


def finite_difference_dt(dt: float, X: np.ndarray) -> np.ndarray:
    dX = np.zeros_like(X)
    dX[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    dX[0] = (X[1] - X[0]) / dt
    dX[-1] = (X[-1] - X[-2]) / dt
    return dX


def evolve_tau_truth(Wi, Lm, eta, L, D, tau, dt):
    trt = np.trace(tau)
    relax = tau + (trt / (Lm**2)) * tau + (trt / (Lm**2)) * np.eye(3)
    dtau_dt = tau @ L + L.T @ tau + 2.0 * eta * D - (1.0 / Wi) * relax
    return sym(tau + dt * dtau_dt)


def simulate_truth(dt, Lt, Wi, Lm, eta, gamma_dot, tau0):
    nt = int(Lt / dt)
    t = np.linspace(0.0, Lt, nt)

    L = make_simple_shear_L(gamma_dot)
    D = sym(L)

    tau = np.zeros((nt, 3, 3), dtype=float)
    tau[0] = sym(tau0)

    for k in trange(nt - 1, desc=f"Sim Wi={Wi:g}, gdot={gamma_dot:g}", leave=False):
        tau[k + 1] = evolve_tau_truth(Wi, Lm, eta, L, D, tau[k], dt)

    D_hist = np.zeros((nt, 3, 3), dtype=float)
    D_hist[:] = D
    return t, tau, D_hist, L, D


def tau1_target(dt, tau, L):
    dtau = finite_difference_dt(dt, tau)
    return dtau - (tau @ L + L.T @ tau)


def make_global_target(Wi, tau1, eta, D_hist):
    return Wi * (tau1 - 2.0 * eta * D_hist)


def component_row_weights(y_train, eps=1e-12, clip_max=10.0):
    std = np.std(y_train, axis=0)
    w = np.zeros_like(std)
    nz = std > eps
    w[nz] = 1.0 / (std[nz] + eps)
    w = w / (np.max(w) + eps)
    w = np.clip(w, 0.0, clip_max)
    return w


def expand_row_weights(w_comp, n_samples):
    return np.tile(w_comp.reshape(9), n_samples)


def normalize_columns(Theta, eps=1e-12):
    s = np.linalg.norm(Theta, axis=0)
    s = np.maximum(s, eps)
    return Theta / s, s


def build_theta(ds: DynamicSINDy, library, D, tau):
    return ds._build_library_matrix(library, D, tau)


def build_weighted_design(ds, library, D_fit, tau_fit, y_fit_scaled):
    """
    Build the weighted, column-normalized design matrix that SINDy uses,
    so BINDy runs on exactly the same features.
    Returns (Theta_n, Yw, scales, sqrtw).
    """
    w_comp = component_row_weights(y_fit_scaled)
    w_rows = expand_row_weights(w_comp, len(D_fit))
    sqrtw = np.sqrt(w_rows + 1e-30)

    Y = y_fit_scaled.reshape(-1)
    Yw = Y * sqrtw

    Theta = build_theta(ds, library, D_fit, tau_fit)
    Theta_w = Theta * sqrtw[:, None]

    Theta_n, scales = normalize_columns(Theta_w)
    return Theta_n, Yw, scales, sqrtw


def hmcmc_generate_tensor_library(N=500, temperature=1.0, max_complexity=3, seed=0):
    np.random.seed(seed)
    sampler = ThermodynamicHMCMC(temperature=temperature, max_complexity=max_complexity)

    I_term = TensorOp("I", lambda D, tau, p: np.eye(3), complexity=1)
    sampler.tensor_library.append(I_term)

    raw = sampler.generate_library(N=N)
    lib = [t for t in raw if isinstance(t, TensorOp)]

    # keep non-rational only (for THIS FENE-P polynomial truth)
    filtered = []
    for term in lib:
        nm = term.name
        if ("c0" in nm) or ("c1" in nm) or ("c2" in nm) or ("/" in nm):
            continue
        filtered.append(term)

    names = {t.name for t in filtered}
    if "τ" not in names:
        filtered.append(TensorOp("τ", lambda D, tau, p: tau, complexity=1))
    if "D" not in names:
        filtered.append(TensorOp("D", lambda D, tau, p: D, complexity=1))
    if "I" not in names:
        filtered.append(I_term)

    seen = set()
    uniq = []
    for t in filtered:
        if t.name in seen:
            continue
        seen.add(t.name)
        uniq.append(t)
    return uniq


def prune_library_by_columns(library, D_sub, tau_sub, y_sub, cos_thr=0.99995, cap=120):
    ds = DynamicSINDy(threshold=0.0, lambda_reg=0.0, max_iter=1)
    Theta = build_theta(ds, library, D_sub, tau_sub)

    w_comp = component_row_weights(y_sub)
    w_rows = expand_row_weights(w_comp, len(D_sub))
    sqrtw = np.sqrt(w_rows + 1e-30)
    Theta_w = Theta * sqrtw[:, None]

    keep = []
    reps = []
    for term, col in zip(library, Theta_w.T):
        nrm = np.linalg.norm(col)
        if nrm < 1e-12:
            continue
        v = col / (nrm + 1e-15)
        dup = False
        for r in reps:
            if abs(np.dot(v, r)) > cos_thr:
                dup = True
                break
        if dup:
            continue
        keep.append(term)
        reps.append(v)
        if len(keep) >= cap:
            break
    return keep


def fit_weighted_sindy(D_fit, tau_fit, y_fit_scaled, library, threshold, lambda_reg):
    ds = DynamicSINDy(threshold=threshold, lambda_reg=lambda_reg, max_iter=20)

    w_comp = component_row_weights(y_fit_scaled)
    w_rows = expand_row_weights(w_comp, len(D_fit))
    sqrtw = np.sqrt(w_rows + 1e-30)

    Y = y_fit_scaled.reshape(-1)
    Yw = Y * sqrtw

    Theta = build_theta(ds, library, D_fit, tau_fit)
    Theta_w = Theta * sqrtw[:, None]

    Theta_n, scales = normalize_columns(Theta_w)
    c_n = ds._sparse_regression(Theta_n, Yw)
    coeffs = c_n / scales

    # >>> KEY FIX 3: softer “active” cutoff for the refit, so ~0.007 survives
    active = np.where(np.abs(coeffs) > threshold / 2.0)[0]
    if len(active) > 0:
        coeffs_refit = np.zeros_like(coeffs)
        coeffs_refit[active] = np.linalg.lstsq(
            Theta_w[:, active], Yw, rcond=None
        )[0]
        coeffs = coeffs_refit

    loss = float(np.mean((Yw - Theta_w @ coeffs) ** 2))
    return coeffs, w_comp, loss


def print_model(coeffs, library, title, thresh=1e-12):
    idx = np.where(np.abs(coeffs) > thresh)[0]
    idx = idx[np.argsort(-np.abs(coeffs[idx]))]
    print(f"\n=== {title} ===")
    if len(idx) == 0:
        print("  (no active terms)")
    for j in idx:
        print(f"{coeffs[j]: .6e} * {library[j].name}")
    return idx


def term_energy_report(D, tau, coeffs_unscaled, library, sample_cap=6000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(D)
    if n > sample_cap:
        ii = rng.choice(n, size=sample_cap, replace=False)
        D0, tau0 = D[ii], tau[ii]
    else:
        D0, tau0 = D, tau

    per = []
    for j, term in enumerate(library):
        c = float(coeffs_unscaled[j])
        if abs(c) < 1e-18:
            per.append((term.name, 0.0))
            continue
        Phi = np.zeros((len(D0), 9))
        for i in range(len(D0)):
            Phi[i] = term.evaluate(D0[i], tau0[i]).reshape(-1)
        contrib = c * Phi
        per.append((term.name, float(np.sqrt(np.mean(contrib**2)))))

    per.sort(key=lambda x: -x[1])
    print("\n=== Term energy (unscaled): RMS contribution ===")
    for name, rms in per:
        print(f"  {rms: .6e}  |  {name}")


# >>> KEY FIX 4: extract FENE-P pieces by behavior (handles name variants)
def match_term_by_behavior(library, D_sub, tau_sub, proto_eval):
    ds = DynamicSINDy(threshold=0.0, lambda_reg=0.0, max_iter=1)
    best_j, best_cos = None, -1.0
    pcol = []
    for i in range(len(D_sub)):
        pcol.append(proto_eval(D_sub[i], tau_sub[i]).reshape(-1))
    pcol = np.concatenate(pcol, axis=0)
    pcol = pcol / (np.linalg.norm(pcol) + 1e-15)

    for j, term in enumerate(library):
        col = []
        for i in range(len(D_sub)):
            col.append(term.evaluate(D_sub[i], tau_sub[i]).reshape(-1))
        col = np.concatenate(col, axis=0)
        nrm = np.linalg.norm(col)
        if nrm < 1e-12:
            continue
        col = col / (nrm + 1e-15)
        cos = abs(float(np.dot(col, pcol)))
        if cos > best_cos:
            best_cos = cos
            best_j = j
    return best_j, best_cos


if __name__ == "__main__":
    all_D, all_tau, all_y = [], [], []

    # -----------------------------------------------------------
    # 1. Generate training data from the truth model
    # -----------------------------------------------------------
    for wi in WI_LIST:
        for gdot in GDOT_LIST:
            for tau0 in TAU0_LIST:
                t, tau, D_hist, L, D = simulate_truth(
                    DT, LT, wi, LM_TRUTH, ETA, gdot, tau0
                )
                tau1 = tau1_target(DT, tau, L)
                y = make_global_target(wi, tau1, ETA, D_hist)

                # skip early transient and downsample
                skip = 30
                idx = np.arange(skip, len(t) - skip, dtype=int)[::3]

                all_D.append(D_hist[idx])
                all_tau.append(tau[idx])
                all_y.append(y[idx])

    D_train = np.concatenate(all_D, axis=0)
    tau_train = np.concatenate(all_tau, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    print("\nTraining samples:", len(D_train))

    y_scale = np.max(np.abs(y_train)) + 1e-30
    y_train_s = y_train / y_scale
    print(f"Target scaling y_scale = {y_scale:.3e}")
    print(
        f"Scaled target RMS = {np.sqrt(np.mean(y_train_s.reshape(-1)**2)):.3e}, "
        f"max|y| = {np.max(np.abs(y_train_s)):.3e}"
    )

    # -----------------------------------------------------------
    # 2. HMCMC candidate library + pruning
    # -----------------------------------------------------------
    print("\n[HMCMC] Generating candidate tensor library...")
    cand = hmcmc_generate_tensor_library(
        N=HMCMC_N_TERMS,
        temperature=HMCMC_TEMPERATURE,
        max_complexity=HMCMC_MAX_COMPLEXITY,
        seed=SEED
    )
    print(f"\nCandidate tensor terms after filtering: {len(cand)}")

    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(
        len(D_train),
        size=min(PRUNE_SUBSAMPLE_SAMPLES, len(D_train)),
        replace=False,
    )
    lib = prune_library_by_columns(
        cand,
        D_train[sub_idx],
        tau_train[sub_idx],
        y_train_s[sub_idx],
        cos_thr=COL_DUP_COS_THR,
        cap=CAP_LIBRARY
    )

    print(f"Library size after pruning/cap: {len(lib)}")
    print("\nLibrary terms (post-prune):")
    for i, term in enumerate(lib[:60]):
        print(f"  {i:3d}. {term.name}  (complexity={term.complexity})")
    if len(lib) > 60:
        print(f"  ... ({len(lib)-60} more)")

    # -----------------------------------------------------------
    # 3. Fit weighted SINDy on a subset
    # -----------------------------------------------------------
    if len(D_train) > FIT_SAMPLE_CAP:
        fit_idx = rng.choice(len(D_train), size=FIT_SAMPLE_CAP, replace=False)
    else:
        fit_idx = np.arange(len(D_train), dtype=int)

    coeffs_s, w_comp, loss = fit_weighted_sindy(
        D_train[fit_idx],
        tau_train[fit_idx],
        y_train_s[fit_idx],
        lib,
        threshold=THRESHOLD,
        lambda_reg=LAMBDA_REG
    )

    print("\n[SINDy] Done. Weighted scaled loss:", loss)

    print_model(coeffs_s, lib, "SINDy model for y (SCALED)")
    coeffs_unscaled = coeffs_s * y_scale
    print_model(coeffs_unscaled, lib, "SINDy model for y (UNSCALED)")
    term_energy_report(D_train, tau_train, coeffs_unscaled, lib,
                       sample_cap=6000, seed=SEED)

    # -----------------------------------------------------------
    # 4. Match SINDy terms to FENE-P structure and recover Lm
    # -----------------------------------------------------------
    Dm = D_train[sub_idx]
    taum = tau_train[sub_idx]

    j_tau, cos_tau = match_term_by_behavior(lib, Dm, taum, lambda D, tau: tau)
    j_ttt, cos_ttt = match_term_by_behavior(
        lib, Dm, taum, lambda D, tau: np.trace(tau) * tau
    )
    j_ti, cos_ti = match_term_by_behavior(
        lib, Dm, taum, lambda D, tau: np.trace(tau) * np.eye(3)
    )

    print("\n[Match by behavior]")
    print(
        f"  tau term match:           idx={j_tau}, "
        f"cos={cos_tau:.6f}, name={lib[j_tau].name}"
    )
    print(
        f"  tr(tau)*tau term match:   idx={j_ttt}, "
        f"cos={cos_ttt:.6f}, name={lib[j_ttt].name}"
    )
    print(
        f"  tr(tau)*I term match:     idx={j_ti},  "
        f"cos={cos_ti:.6f},  name={lib[j_ti].name}"
    )

    ctau = float(coeffs_unscaled[j_tau])
    ctt = float(coeffs_unscaled[j_ttt])
    cti = float(coeffs_unscaled[j_ti])

    invLm2 = 0.5 * (abs(ctt) + abs(cti))
    if invLm2 > 0:
        Lm_hat = float(np.sqrt(1.0 / invLm2))
        print("\n=== Recovered compact FENE-P coefficients (SINDy) ===")
        print(f"  coeff(tau)           ≈ {ctau:.6f}   (truth -1)")
        print(
            f"  coeff(tr(tau)*tau)   ≈ {ctt:.6f}   "
            f"(truth -1/Lm^2 = -0.04)"
        )
        print(
            f"  coeff(tr(tau)*I)     ≈ {cti:.6f}   "
            f"(truth -1/Lm^2 = -0.04)"
        )
        print(f"  => inferred 1/Lm^2   ≈ {invLm2:.6f}")
        print(f"  => inferred Lm       ≈ {Lm_hat:.6f}   (truth {LM_TRUTH})")
    else:
        print(
            "\n[WARN] Could not infer Lm (trace-term coefficients too "
            "small/zero). Lower THRESHOLD further."
        )

    # -----------------------------------------------------------
    # 5. BINDy: RJMCMC on same weighted/normalized design
    # -----------------------------------------------------------
    print("\n[BINDy] Running RJMCMC on same weighted/normalized design...")

    ds_tmp = DynamicSINDy(threshold=0.0, lambda_reg=0.0, max_iter=1)
    D_fit = D_train[fit_idx]
    tau_fit = tau_train[fit_idx]
    y_fit_scaled = y_train_s[fit_idx]

    Theta_n, Yw, scales, sqrtw = build_weighted_design(
        ds_tmp, lib, D_fit, tau_fit, y_fit_scaled
    )

    D_terms = Theta_n.shape[1]
    term_complexities = [t.complexity for t in lib]

    mu0 = np.zeros(D_terms)
    Sigma0_diag = 1e3 * np.ones(D_terms)

    bindy = BINDyRJ(
        Theta_n,
        Yw,
        mu0=mu0,
        Sigma0_diag=Sigma0_diag,
        term_complexity=term_complexities,
        a0=1e-3,
        b0=1e-3,
        model_prior="geo_complexity",
        theta_geom=0.98,
        complexity_weight=1.0,
        n_steps=6000,
        burn_in=1000,
        min_active=3,
        seed=SEED,
    )

    masks, sigma2s = bindy.sample(init_model="full", init_sigma2=1.0)

    inclusion_probs = masks.mean(axis=0)
    print("\n=== BINDy term inclusion probabilities ===")
    for j, term in enumerate(lib):
        print(f"{j:3d}: p={inclusion_probs[j]:.3f}   {term.name}")

    coeffs_bindy_norm = bindy.compute_posterior_mean_xi(masks, sigma2s)
    coeffs_bindy_weighted = coeffs_bindy_norm / scales
    coeffs_bindy_unscaled = coeffs_bindy_weighted * y_scale

    print_model(
        coeffs_bindy_unscaled,
        lib,
        "BINDy model for y (UNSCALED)"
    )
    term_energy_report(
        D_train,
        tau_train,
        coeffs_bindy_unscaled,
        lib,
        sample_cap=6000,
        seed=SEED,
    )

    # Recover FENE-P parameters from BINDy model
    ctau_B = float(coeffs_bindy_unscaled[j_tau])
    ctt_B = float(coeffs_bindy_unscaled[j_ttt])
    cti_B = float(coeffs_bindy_unscaled[j_ti])

    invLm2_B = 0.5 * (abs(ctt_B) + abs(cti_B))
    if invLm2_B > 0:
        Lm_hat_B = float(np.sqrt(1.0 / invLm2_B))
        print("\n=== Recovered compact FENE-P from BINDy ===")
        print(f"  coeff(tau)           ≈ {ctau_B:.6f}   (truth -1)")
        print(
            f"  coeff(tr(tau)*tau)   ≈ {ctt_B:.6f}   "
            f"(truth -1/Lm^2 = -0.04)"
        )
        print(
            f"  coeff(tr(tau)*I)     ≈ {cti_B:.6f}   "
            f"(truth -1/Lm^2 = -0.04)"
        )
        print(f"  => inferred 1/Lm^2   ≈ {invLm2_B:.6f}")
        print(f"  => inferred Lm       ≈ {Lm_hat_B:.6f}   (truth {LM_TRUTH})")
    else:
        print(
            "\n[BINDy] Could not infer Lm (trace-term coefficients too small)."
        )

    # -----------------------------------------------------------
    # 6. Save SINDy results 
    # -----------------------------------------------------------
    np.savez_compressed(
        os.path.join(OUTDIR, "hmcmc_fenep_discovery_v2.npz"),
        dt=DT,
        Lt=LT,
        eta=ETA,
        Lm_truth=LM_TRUTH,
        Wi_list=np.array(WI_LIST),
        gdot_list=np.array(GDOT_LIST),
        y_scale=y_scale,
        coeffs_scaled=coeffs_s,
        coeffs_unscaled=coeffs_unscaled,
        library_names=np.array([t.name for t in lib], dtype=object),
        row_weights=w_comp,
        loss=loss,
        threshold=THRESHOLD,
    )
    print(f"\nSaved {OUTDIR}/hmcmc_fenep_discovery_v2.npz")

