# This class will set up the optimization problem for Spherical Tutte's Embeddings
# Again referencing Algorithm 1 - Modified L-BFGS two loop recursion from the paper

# Libraries
import numpy as np
import scipy as sp
import torch
from torch.optim import LBFGS
from torch.linalg import qr
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle


class SOTESolver:
    def __init__(self,V, F, x0, N, w, A, L):

        # x0 - 7662 x1 - column stack of phi - vertex points on the sphere
        # A - 165 x 7662
        # N - 7662 x 7469 - QR decomposition
        # w - 2554 x 2554 Laplacian
        # self.N_im = torch.tensor(N)
        self.A = torch.tensor(A)
        print(A.shape)
        self.x0 = torch.tensor(x0)
        self.x = torch.tensor(x0, requires_grad = True)
        self.w = torch.tensor(w)
        self.L = torch.tensor(L)
        print(self.is_laplacian(self.w))
        self.V = torch.tensor(V)
        self.F = torch.tensor(F)
        # L_check = self.compute_cotangent_laplacian(V, F)
        # print(self.is_laplacian(L_check))

        # QR decomposition of A transpose
        # Perform SVD on A transpose
        U, S, Vh = torch.linalg.svd(self.A.T)  # Full SVD
        rank = torch.sum(S > 1e-10).item()  # Numerical rank of A transpose

        # Null space basis
        self.N = U[:, rank:]  # Columns of U corresponding to zero singular values
        print("N shape (Null space):", self.N.shape)


        self.m = 3
        self.maxiter = 100
        self.tolX = 1e-10
        self.tolFun = 0

        self.optimizer_Init()


    def optimizer_Init(self):

        # Initialize the supplementary vars - size (N x m)
        self.s_k = torch.zeros((self.N.shape[1], self.m))
        self.y_k = torch.zeros((self.N.shape[1], self.m))
        self.rao_k = torch.zeros((1, self.m))
        self.a_k = torch.zeros((1, self.m))
        self.bet_k = torch.zeros((1, self.m))

        Q = torch.kron(self.w.contiguous(), torch.eye(3))
        self.d_energy = torch.matmul(self.x.T, torch.matmul(Q, self.x))

        # Initialize 
        # Initialize Energy

        # define pytorch instance of lbfgs optimizer
        self.optimizer = LBFGS(params = [self.x], lr = 0.001, max_iter = 50, tolerance_grad = 1e-5, line_search_fn="strong_wolfe")


    def updateMemory(self):
        # Update of variables after the initial run
        self.s_k = torch.matmul(self.N.T ,(self.x - self.x_prev))
        self.y_k = torch.matmul(self.N.T , (self.x_grad - self.x_grad_prev))
        self.rao_k = 1 / torch.matmul(self.y_k.T, self.s_k)

    def verify(self):
        assert torch.allclose(self.A @ self.N, torch.zeros_like(self.A @ self.N), atol=1e-6), "N is not a null space basis!"

    def visualize_plot(self):
        # Pre-Process x0
        #reshape
        x = np.reshape(self.x.detach().numpy(), (-1,3))
        print(x.shape)
        X = x[:,0]
        Y = x[:,1]
        Z = x[:,2]
        print(X.shape)

        # as part of debug - plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z)
        plt.show()

        return


    def closure(self):
        self.optimizer.zero_grad()

        # compute cotan 
        # Double check Laplacian Matrix
        Q = torch.kron(self.w.contiguous(), torch.eye(3))
        # Compute Dirichlet energy
        self.d_energy = -torch.matmul(torch.matmul(self.x.T,Q), self.x)

        # calc denergy/dx - looking 
        self.d_energy.backward()
    
        # Boundary Constraints
        # Update the gradient to the projected gradient
        proj_grad = self.N @ (self.N.T @ self.x.grad)

        with torch.no_grad():
            self.x.grad[:] = proj_grad

        return self.d_energy



    def custom_opt_loop(self):
        # define Q
        self.q = torch.matmul(self.N.T, self.x_fgrad)

        # Main Loop - need to initialize the starting values of s, y, 
        for i in self.m:
            self.a_k[i] = torch.matmul(torch.matmul(self.rao_k[i], self.s_k[:,i].T), self.q)
            self.q -= torch.matmul(self.a_k[i], self.y_k[:,i])

        self.gamma_k = torch.divide(torch.matmul(self.s_k.T, self.y_k), torch.matmul(self.y_k.T, self.y_k))
        l = torch.kron(self.w, torch.eye(3))
        energy = torch.matmul(torch.matmul(self.N.T, l), self.N)
        self.r = torch.matmul(torch.matmul(self.gamma_k, torch.inverse(energy)), self.q)

        for ii in self.m:
            self.bet_k[ii] = torch.matmul(torch.matmul(self.rao_k[ii], self.y_k.T[:,ii]), self.r)
            self.r += torch.matmul(self.s_k[:,ii], (self.a_k - self.bet_k))

        self.p = torch.matmul(self.N, self.r)

        # Parameter for line search - need to develop method after
        self.t = 1

        # Update positions of vertices X 
        self.x = self.x + torch.matmul(self.t*self.p)

        # Update Memory 
        self.updateMemory()


    def is_laplacian(self, mat):
        """
        Checks if the given matrix is a Laplacian matrix.
        
        Args:
            matrix (torch.Tensor): The input square matrix of size (n, n).
            
        Returns:
            bool: True if the matrix is a Laplacian, False otherwise.
        """
        matrix = mat
        # Check if the matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        # Check if the matrix is symmetric
        if not torch.allclose(matrix, matrix.T):
            return False
        print("Symmetric")

        # Check diagonal entries are the sum of the absolute values of off-diagonal entries
        for i in range(matrix.shape[0]):
            # Sum of off-diagonal elements in the i-th row
            off_diagonal_sum = torch.sum(torch.abs(matrix[i])) - torch.abs(matrix[i, i])
            if not torch.isclose(abs(matrix[i, i]), off_diagonal_sum, atol=1e-6):
                return False
        
        print("Diagonals are Good")

        # Check if all off-diagonal elements are non-positive
        if not torch.all(matrix - torch.diag(torch.diagonal(matrix)) < 0):
            return False
        
        print("Off diagonal elements are non-positive")

        return True

    def compute_cotangent_laplacian(self,vertices, faces):
        """
        Computes the cotangent Laplacian matrix for a 3D mesh.

        Args:
            vertices (torch.Tensor): Tensor of shape (n, 3) containing vertex positions.
            faces (torch.Tensor): Tensor of shape (m, 3) containing face vertex indices.

        Returns:
            L (torch.Tensor): Sparse cotangent Laplacian matrix of shape (n, n).
        """
        print(vertices.shape)
        print(faces.shape)

        # Step 1: Extract vertex positions for each face
        v0 = vertices[faces[:, 0]]  # Vertices at index 0 of each face
        v1 = vertices[faces[:, 1]]  # Vertices at index 1 of each face
        v2 = vertices[faces[:, 2]]  # Vertices at index 2 of each face

        # Step 2: Compute edge vectors
        e0 = v2 - v1  # Edge opposite v0
        e1 = v0 - v2  # Edge opposite v1
        e2 = v1 - v0  # Edge opposite v2

        # Step 3: Compute cotangents for each angle in the triangle
        def cotangent(u, v):
            return torch.sum(u * v, dim=1) / torch.linalg.norm(torch.cross(u, v), dim=1)

        cot0 = cotangent(e1, e2)  # Cotangent at vertex 0
        cot1 = cotangent(e2, e0)  # Cotangent at vertex 1
        cot2 = cotangent(e0, e1)  # Cotangent at vertex 2

        # Step 4: Accumulate weights for each edge
        n_vertices = vertices.shape[0]
        I = torch.cat([faces[:, [1, 2]], faces[:, [2, 0]], faces[:, [0, 1]]], dim=0).flatten()
        J = torch.cat([faces[:, [2, 0]], faces[:, [0, 1]], faces[:, [1, 2]]], dim=0).flatten()
        W = torch.cat([cot0, cot1, cot2, cot0, cot1, cot2], dim=0) * 0.5

        # Create sparse matrix for weights
        L = torch.sparse_coo_tensor(
            torch.stack([I, J], dim=0),
            -W,
            size=(n_vertices, n_vertices)
        )

        # Step 5: Compute diagonal entries (row-sum)
        diagonal = torch.sparse.sum(L, dim=1).to_dense()
        L = L + torch.sparse_coo_tensor(
            torch.stack([torch.arange(n_vertices), torch.arange(n_vertices)]),
            diagonal,
            size=(n_vertices, n_vertices)
        )

        return L

    def plotEnergy(self, step, energy):
        fig, ax = plt.subplots(figsize = (10,10))
        ax = sns.lineplot(x = step, y = energy)

    def solve(self):
        # self.verify()
        stepArr = []
        energyArr = []
        # Main loop to iterate through LBFGS calculations
        for i in range(10):
            if i == 0 or  i ==9:
                self.visualize_plot()
            energy = self.optimizer.step(self.closure)
            print(f"Step {i}, Grad: {self.x.grad}, Energy: {energy}")
            stepArr.append(i)
            energyArr.append(energy)
        # self.plotEnergy(stepArr, energyArr)


