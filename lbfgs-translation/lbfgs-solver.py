import numpy as np
from time import time
from dataclasses import dataclass

@dataclass
class LinearConstraints:
    A: np.ndarray  # Constraint matrix
    b: np.ndarray  # Constraint vector
    N: np.ndarray  # Null space basis
    dim_N: int     # Dimension of null space

class SimplePreconditioner:
    def apply(self, q, x):
        return q  # Identity preconditioner for testing

class OptimSolverLBFGS:
    def __init__(self, fun, x0, Ab, P, m):
        self.fun = fun
        self.x0 = x0.copy()
        self.Ab = Ab
        self.P = P
        self.m = m
        
        # Initialize state
        self.x = self.x0
        self.x_prev = None
        self.f_count = 0
        self.grad_count = 0
        
        # L-BFGS memory
        self.curr_m = 0
        self._init_memory()
        
        # Constants
        self.tol_Ab = 1e-10
        self.ls_alpha = 0.2
        self.ls_beta = 0.5
        
        # Initial evaluation
        self.x_f, self.x_fgrad = self._evaluate(self.x)
    
    def _init_memory(self):
        """Initialize L-BFGS memory matrices"""
        self.y_k = np.zeros((self.Ab.dim_N, self.m))
        self.s_k = np.zeros((self.Ab.dim_N, self.m))
        self.rao_k = np.zeros(self.m)
        self.alpha_k = np.zeros(self.m)
        self.beta_k = np.zeros(self.m)
        self.y_k[:,0] = self.s_k[:,0] = 1  # Initial H_k
    
    def _evaluate(self, x):
        """Evaluate function and gradient"""
        f, grad = self.fun(x)
        self.f_count += 1
        self.grad_count += 1
        return f, grad
    
    def iterate(self):
        """Perform one L-BFGS iteration"""
        self.x_prev = self.x.copy()
        self.x_fgrad_prev = self.x_fgrad.copy()
        
        # Two-loop recursion
        q = self.Ab.N.T @ self.x_fgrad
        for i in range(self.curr_m):
            self.alpha_k[i] = self.rao_k[i] * (self.s_k[:,i] @ q)
            q -= self.alpha_k[i] * self.y_k[:,i]
        
        # Initial Hessian approximation
        gamma = (self.s_k[:,0] @ self.y_k[:,0]) / (self.y_k[:,0] @ self.y_k[:,0])
        r = gamma * self.P.apply(q, self.x)
        
        for i in range(self.curr_m):
            self.beta_k[i] = self.rao_k[i] * (self.y_k[:,i] @ r)
            r += self.s_k[:,i] * (self.alpha_k[i] - self.beta_k[i])
        
        # Update position
        p = -self.Ab.N @ r
        t = 1.0
        
        # Simple backtracking line search
        while True:
            x_new = self.x + t * p
            f_new = self.fun(x_new)[0]
            self.f_count += 1
            if f_new <= self.x_f + self.ls_alpha * t * (self.x_fgrad @ p):
                break
            t *= self.ls_beta
        
        self.x = x_new
        self.x_f, self.x_fgrad = self._evaluate(self.x)
        
        # Update L-BFGS memory
        self.y_k[:,1:] = self.y_k[:,:-1]
        self.s_k[:,1:] = self.s_k[:,:-1]
        self.rao_k[1:] = self.rao_k[:-1]
        
        self.curr_m = min(self.curr_m + 1, self.m)
        self.y_k[:,0] = self.Ab.N.T @ (self.x_fgrad - self.x_fgrad_prev)
        self.s_k[:,0] = self.Ab.N.T @ (self.x - self.x_prev)
        self.rao_k[0] = 1.0 / (self.y_k[:,0] @ self.s_k[:,0])
    
    def solve(self, max_iter=100, tol=1e-6):
        """Solve optimization problem"""
        for _ in range(max_iter):
            self.iterate()
            if np.linalg.norm(self.x - self.x_prev) < tol:
                break
        return self.x, self.x_f

def test_lbfgs_solver():
    """Test the L-BFGS solver with a simple quadratic objective"""
    # Problem setup
    n, m = 100, 10  # dimension and constraints
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    
    # Compute null space basis
    Q, R = np.linalg.qr(A.T, mode='complete')
    N = Q[:,m:]  # null space basis
    
    # Create constraints object
    Ab = LinearConstraints(A=A, b=b, N=N, dim_N=n-m)
    
    # Particular solution
    x_particular = A.T @ np.linalg.solve(A @ A.T, b)
    
    # Initial point
    x0 = x_particular + N @ np.random.randn(n-m)
    
    # Objective function (simple quadratic)
    def objective(x):
        f = 0.5 * np.sum(x**2)
        grad = x
        return f, grad
    
    # Create and run solver
    solver = OptimSolverLBFGS(objective, x0, Ab, SimplePreconditioner(), m=5)
    x, f = solver.solve()
    
    return solver

if __name__ == "__main__":
    solver = test_lbfgs_solver()
    print(f"Final objective value: {solver.x_f:.6f}")
    print(f"Function evaluations: {solver.f_count}")
