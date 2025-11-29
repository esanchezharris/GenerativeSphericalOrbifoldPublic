import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import unittest

class SparseLU:
    """
    Sparse LU decomposition solver using scipy.sparse.linalg.
    Stores LU decomposition of a sparse matrix and provides methods to solve linear systems.
    """
    
    def __init__(self, LHS: sparse.spmatrix):
        """
        Initialize SparseLU with a sparse matrix.
        
        Args:
            LHS: Sparse matrix to decompose (left-hand side matrix)
            
        Raises:
            TypeError: If input matrix is not sparse
        """
        if not sparse.issparse(LHS):
            raise TypeError('Argument must be a sparse matrix')
            
        self.LHS = LHS
        # scipy.sparse.linalg.splu performs LU decomposition
        # Note: scipy handles permutations internally, different from MATLAB
        self.lu = splu(LHS)
        
    def solve(self, RHS: np.ndarray) -> np.ndarray:
        """
        Solve the linear system LHS * x = RHS using the stored LU decomposition.
        
        Args:
            RHS: Right-hand side vector or matrix
            
        Returns:
            Solution of the linear system
            
        Note: 
            scipy.sparse.linalg.splu.solve handles the permutations and
            forward/backward substitutions internally
        """
        return self.lu.solve(RHS)


class TestSparseLU(unittest.TestCase):
    def setUp(self):
        """Create test matrices for each test case"""
        # Create a sparse test matrix
        n = 5
        data = np.array([2., -1., -1., 2., -1., -1., 2., -1., -1., 2.])
        row = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
        col = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3])
        self.A = sparse.coo_matrix((data, (row, col)), shape=(n-1, n-1)).tocsc()
        
        # Create a dense right-hand side
        self.b = np.ones(n-1)
        
        # Create solver
        self.solver = SparseLU(self.A)
        
    def test_init_with_dense_matrix(self):
        """Test that initialization with dense matrix raises TypeError"""
        dense_matrix = np.array([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            SparseLU(dense_matrix)
            
    def test_solve_single_rhs(self):
        """Test solving with single right-hand side"""
        x = self.solver.solve(self.b)
        
        # Check that solution satisfies the original system
        residual = np.linalg.norm(self.A @ x - self.b)
        self.assertLess(residual, 1e-10)
        
    def test_solve_multiple_rhs(self):
        """Test solving with multiple right-hand sides"""
        B = np.column_stack([self.b, 2*self.b, 3*self.b])
        X = self.solver.solve(B)
        
        # Check each solution
        for i in range(3):
            residual = np.linalg.norm(self.A @ X[:, i] - B[:, i])
            self.assertLess(residual, 1e-10)
            
    def test_solve_different_shapes(self):
        """Test solving with different RHS shapes"""
        # Test with column vector
        b_col = self.b.reshape(-1, 1)
        x_col = self.solver.solve(b_col)
        self.assertEqual(x_col.shape, b_col.shape)
        
        # Test with row vector
        b_row = self.b.reshape(1, -1)
        with self.assertRaises(ValueError):
            # Should raise error because dimensions don't match
            self.solver.solve(b_row)
            
    def test_reproducibility(self):
        """Test that multiple solves with same RHS give same result"""
        x1 = self.solver.solve(self.b)
        x2 = self.solver.solve(self.b)
        np.testing.assert_array_almost_equal(x1, x2)
        
    def test_system_accuracy(self):
        """Test accuracy of the solution for a known system"""
        # Create a system with known solution
        n = 3
        diagonals = np.array([1, 2, 3])  # Values on the main diagonal
        A = sparse.diags(diagonals, format='csc')  # Creates a diagonal matrix
        b = np.array([1, 2, 3])
        expected_x = b / diagonals  # For diagonal matrix, x = b/diagonal
        
        solver = SparseLU(A)
        x = solver.solve(b)
        
        np.testing.assert_array_almost_equal(x, expected_x)

if __name__ == '__main__':
    unittest.main()
