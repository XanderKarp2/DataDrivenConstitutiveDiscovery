import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from sampling import ThermodynamicHMCMC

class DynamicSINDy:
    """
    SINDy that learns constants as terms become active
    """
    
    def __init__(self, threshold=0.05, lambda_reg=0.01, max_iter=20):
        self.threshold = threshold
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
    
    def fit(self, D, tau, UCD, library):
        """
        Iterative SINDy with constant optimization:
        1. Sparse regression to find active terms
        2. For active terms with params, optimize those params
        3. Re-run sparse regression
        4. Repeat until convergence
        """
        Y = UCD.reshape(-1)
        n_samples = len(D)
        
        print(f"\n[SINDy] Starting dynamic optimization...")

        best_loss = float('inf')
        best_coeffs = None
        best_library = None
        
        # Make working copy of library
        working_library = [deepcopy(term) for term in library]
        
        for iteration in range(self.max_iter):
            # Build library matrix with current parameters
            Theta = self._build_library_matrix(working_library, D, tau)
            
            # Sparse regression
            coeffs = self._sparse_regression(Theta, Y)
            
            # Compute loss
            Y_pred = Theta @ coeffs
            loss = np.mean((Y - Y_pred)**2)
            
            # Find active terms
            active_mask = np.abs(coeffs) > self.threshold / 2
            active_idx = np.where(active_mask)[0]
            n_active = len(active_idx)
            
            print(f"  Iter {iteration+1:2d}: Loss={loss:.6e}, Active={n_active:2d} terms", end="")
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                best_coeffs = coeffs.copy()
                best_library = [deepcopy(term) for term in working_library]
            
            # Check convergence
            if iteration > 0 and abs(loss - prev_loss) / (prev_loss + 1e-10) < 1e-6:
                print(" → Converged!")
                break
            
            prev_loss = loss
            
            # Optimize constants for active parameterized terms
            param_terms = [j for j in active_idx if len(working_library[j].params) > 0]
            
            if param_terms:
                print(f" → Optimizing {len(param_terms)} param terms...")
                
                for j in param_terms:
                    term = working_library[j]
                    
                    # Optimize this term's constants
                    def loss_fn(params_new):
                        term_temp = deepcopy(term)
                        term_temp.params = params_new
                        
                        # Rebuild just this term's contribution
                        library_temp = working_library.copy()
                        library_temp[j] = term_temp
                        
                        Theta_temp = self._build_library_matrix(library_temp, D, tau)
                        coeffs_temp = self._sparse_regression(Theta_temp, Y)
                        Y_pred_temp = Theta_temp @ coeffs_temp
                        
                        return np.mean((Y - Y_pred_temp)**2)
                    
                    # Optimize
                    result = minimize(
                        loss_fn,
                        term.params,
                        method='Powell',
                        bounds=term.param_bounds if term.param_bounds else None,
                        options={'maxiter': 10, 'ftol': 1e-6}
                    )
                    
                    if result.success and result.fun < loss:
                        working_library[j].params = result.x
                        print(f"      Term {j} ({term.name[:30]}...): params={result.x}")
            else:
                print()
        
        return best_coeffs, best_library, best_loss
    
    def _build_library_matrix(self, library, D, tau):
        """Build Theta matrix"""
        n_samples = len(D)
        n_terms = len(library)
        Theta = np.zeros((n_samples * 9, n_terms))
        
        for j, term in enumerate(library):
            for i in range(n_samples):
                try:
                    result = term.evaluate(D[i], tau[i])
                    Theta[i*9:(i+1)*9, j] = result.flatten()
                except:
                    # If evaluation fails, leave as zeros
                    pass
        
        return Theta
    
    def _sparse_regression(self, Theta, Y):
        """Standard STRidge"""
        m = Theta.shape[1]
        coeffs = np.linalg.lstsq(Theta, Y, rcond=None)[0]
        
        for _ in range(10):
            # Regularized regression
            coeffs = np.linalg.lstsq(
                Theta.T @ Theta + self.lambda_reg * np.eye(m),
                Theta.T @ Y,
                rcond=None
            )[0]
            
            # Threshold small coefficients
            small = np.abs(coeffs) < self.threshold
            coeffs[small] = 0
            
            # Re-fit active terms
            active = ~small
            if np.sum(active) == 0:
                break
            
            coeffs[active] = np.linalg.lstsq(Theta[:, active], Y, rcond=None)[0]
        
        return coeffs