import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union

# ------------------------
# Primitive operation types
# ------------------------

@dataclass
class TensorOp:
    name: str
    func: Callable        # (D, tau, params) -> tensor
    complexity: int
    params: np.ndarray = field(default_factory=lambda: np.array([]))
    param_bounds: List[Tuple[float, float]] = field(default_factory=list)
    type: str = "tensor"  # all ops have type: 'tensor'

    def evaluate(self, D, tau):
        return self.func(D, tau, self.params)


@dataclass
class ScalarOp:
    name: str
    func: Callable        # (D, tau, params) -> scalar
    complexity: int
    params: np.ndarray = field(default_factory=lambda: np.array([]))
    param_bounds: List[Tuple[float, float]] = field(default_factory=list)
    type: str = "scalar"  # all ops have type: 'scalar'

    def evaluate(self, D, tau):
        return self.func(D, tau, self.params)


# ------------------------
# HMCMC Sampler
# ------------------------

class ThermodynamicHMCMC:
    def __init__(self, temperature=1.0, max_complexity=4):
        self.temperature = temperature
        self.max_complexity = max_complexity

        # Base primitives
        self.base_tensors: List[TensorOp] = [
            TensorOp("D", lambda D, tau, p: D, 1),
            TensorOp("τ", lambda D, tau, p: tau, 1),
        ]
        self.base_scalars: List[ScalarOp] = [
            ScalarOp(f"c{i}", lambda D, tau, p: p[i], 1) for i in range(3)
        ]

        # Libraries
        self.tensor_library: List[TensorOp] = self.base_tensors.copy()
        self.scalar_library: List[ScalarOp] = self.base_scalars.copy()

    def generate_library(self, N=30):
        """Generate symbolic library using HMCMC"""
        library = self.tensor_library.copy()
        attempts = 0
        max_tries = N * 20

        while len(library) < N and attempts < max_tries:
            attempts += 1
            expr = self._propose_expression()
            if expr is None:
                continue

            # Avoid duplicates
            if any(t.name == expr.name for t in library):
                continue

            # Metropolis acceptance based on complexity
            delta_complexity = expr.complexity - np.mean([t.complexity for t in library])
            prob = np.exp(-delta_complexity / self.temperature)
            if np.random.rand() < prob:
                library.append(expr)
                if expr.type == "tensor":
                    self.tensor_library.append(expr)
                else:
                    self.scalar_library.append(expr)
                print(f"{len(library):2d}. {expr.name:<40s} (complexity={expr.complexity})")

        return library

    # ------------------------
    # Expression Proposals
    # ------------------------
    def _propose_expression(self) -> Union[TensorOp, ScalarOp, None]:
        """Propose a new tensor or scalar expression generatively"""
        # Randomly choose target type
        target_type = np.random.choice(["tensor", "scalar"], p=[0.7, 0.3])

        if target_type == "tensor":
            return self._propose_tensor_expression()
        else:
            return self._propose_scalar_expression()

    # ------------------------
    # Tensor expressions
    # ------------------------
    def _propose_tensor_expression(self):
        if len(self.tensor_library) < 1:
            return None

        # Randomly choose operation
        op_type = np.random.choice(["product", "scale_trace", "rational"], p=[0.4, 0.3, 0.3])

        if op_type == "product":
            return self._tensor_product()
        elif op_type == "scale_trace":
            return self._tensor_tr_scale()
        else:
            return self._tensor_rational()

    def _tensor_product(self):
        if len(self.tensor_library) < 2:
            return None
        A, B = np.random.choice(self.tensor_library, size=2, replace=True)
        if A is None or B is None:
            return None
        complexity = A.complexity + B.complexity
        if complexity > self.max_complexity:
            return None
        name = f"({A.name}@{B.name})"
        func = lambda D, tau, p: A.evaluate(D, tau) @ B.evaluate(D, tau)
        return TensorOp(name, func, complexity)

    def _tensor_tr_scale(self):
        if len(self.tensor_library) < 2:
            return None
        A, B = np.random.choice(self.tensor_library, size=2, replace=True)
        if A is None or B is None:
            return None
        complexity = A.complexity + B.complexity
        if complexity > self.max_complexity:
            return None
        name = f"tr({A.name})*{B.name}"
        func = lambda D, tau, p: np.trace(A.evaluate(D, tau)) * B.evaluate(D, tau)
        return TensorOp(name, func, complexity)

    def _tensor_rational(self):
        if len(self.tensor_library) < 1 or len(self.scalar_library) < 1:
            return None
        A = np.random.choice(self.tensor_library)
        B = np.random.choice(self.tensor_library)
        if A is None or B is None:
            return None
        complexity = A.complexity + B.complexity + 1
        if complexity > self.max_complexity:
            return None
        # Learnable params
        params = np.array([1.0, 100.0, 1.0])
        bounds = [(0.1, 10.0), (10.0, 1000.0), (0.1, 10.0)]
        name = f"(c0*{A.name})/(c1 - c2*tr({B.name}))"
        func = lambda D, tau, p: (p[0]*A.evaluate(D, tau)) / max(p[1]-p[2]*np.trace(B.evaluate(D, tau)), 0.1)
        return TensorOp(name, func, complexity, params, bounds)

    # ------------------------
    # Scalar expressions
    # ------------------------
    def _propose_scalar_expression(self):
        if len(self.scalar_library) < 1:
            return None
        # Combine scalars or take trace of tensor
        op_type = np.random.choice(["sum", "ratio", "trace"], p=[0.4,0.3,0.3])
        if op_type == "sum":
            A, B = np.random.choice(self.scalar_library, size=2, replace=True)
            if A is None or B is None:
                return None
            complexity = A.complexity + B.complexity
            name = f"({A.name}+{B.name})"
            func = lambda D, tau, p: A.evaluate(D, tau) + B.evaluate(D, tau)
            return ScalarOp(name, func, complexity)
        elif op_type == "ratio":
            A, B = np.random.choice(self.scalar_library, size=2, replace=True)
            if A is None or B is None:
                return None
            complexity = A.complexity + B.complexity
            name = f"({A.name}/{B.name})"
            func = lambda D, tau, p: A.evaluate(D, tau) / max(B.evaluate(D, tau), 1e-6)
            return ScalarOp(name, func, complexity)
        else:  # trace of a tensor
            if len(self.tensor_library) < 1:
                return None
            T = np.random.choice(self.tensor_library)
            if T is None:
                return None
            complexity = T.complexity + 1
            name = f"tr({T.name})"
            func = lambda D, tau, p: np.trace(T.evaluate(D, tau))
            return ScalarOp(name, func, complexity)
