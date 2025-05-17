"""
ProNE: Graph Embedding via Spectral Propagation
------------------------------------------------
This module implements the ProNE algorithm for node embedding based on
randomized SVD and Chebyshev polynomial spectral enhancement.

Reference:
Jie Zhang, Yuxiao Dong, Yan Wang, Jie Tang and Ming Ding.
"ProNE: Fast and Scalable Network Representation Learning"
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd


class ProNE:
    """
    ProNE Graph Embedding Model

    Parameters
    ----------
    G : networkx.Graph
        Input graph (should be undirected or will be converted).
    emb_size : int, default=128
        Dimensionality of the node embeddings.
    step : int, default=10
        Propagation steps for Chebyshev approximation.
    theta : float, default=0.5
        Gaussian kernel parameter.
    mu : float, default=0.2
        Chebyshev polynomial shift.
    n_iter : int, default=5
        Number of iterations for randomized SVD.
    random_state : int, default=2023
        Random seed.
    """

    def __init__(self, G, emb_size=128, step=10, theta=0.5, mu=0.2, n_iter=5, random_state=2023):
        self.G = G.to_undirected()
        self.emb_size = emb_size
        self.node_number = self.G.number_of_nodes()
        self.step = step
        self.theta = theta
        self.mu = mu
        self.n_iter = n_iter
        self.random_state = random_state

        # Create adjacency matrix
        mat = scipy.sparse.lil_matrix((self.node_number, self.node_number))
        for u, v in self.G.edges():
            if u != v:
                mat[int(u), int(v)] = 1
                mat[int(v), int(u)] = 1
        self.mat = scipy.sparse.csr_matrix(mat)

    def get_embedding_rand(self, matrix):
        """Generate sparse embedding using randomized SVD."""
        t1 = time.time()
        smat = scipy.sparse.csc_matrix(matrix)
        U, Sigma, _ = randomized_svd(
            smat, n_components=self.emb_size, n_iter=self.n_iter, random_state=self.random_state
            )
        U = np.nan_to_num(U.astype(float)) * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print('Randomized SVD time:', time.time() - t1)
        return U

    def get_embedding_dense(self, matrix, emb_size):
        """Generate dense embedding using full SVD."""
        t1 = time.time()
        U, s, _ = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
        U = U[:, :emb_size] * np.sqrt(s[:emb_size])
        U = preprocessing.normalize(U, "l2")
        print('Dense SVD time:', time.time() - t1)
        return U

    def fit(self, tran, mask):
        """
        Fit model to graph via PMI approximation and randomized SVD.

        Parameters
        ----------
        tran : scipy.sparse matrix
            Co-occurrence transition matrix.
        mask : scipy.sparse matrix
            Mask matrix used for negative sampling.

        Returns
        -------
        np.ndarray
            Initial node embeddings.
        """
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1
        neg = scipy.sparse.diags(neg / neg.sum(), format="csr")
        neg = mask.dot(neg)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)
        C1 -= neg

        return self.get_embedding_rand(C1)

    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        """
        Enhance embeddings using spectral propagation via Chebyshev polynomials.

        Parameters
        ----------
        A : scipy.sparse matrix
            Adjacency matrix.
        a : np.ndarray
            Initial embeddings.
        order : int, default=10
            Polynomial order.
        mu : float, default=0.5
            Shift parameter.
        s : float, default=0.5
            Scale parameter for Gaussian kernel.

        Returns
        -------
        np.ndarray
            Refined node embeddings.
        """
        if order == 1:
            return a

        A = sp.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = sp.eye(self.node_number) - DA
        M = L - mu * sp.eye(self.node_number)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0 - 2 * iv(1, s) * Lx1

        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            conv += (2 * iv(i, s) * Lx2) if i % 2 == 0 else (-2 * iv(i, s) * Lx2)
            Lx0, Lx1 = Lx1, Lx2

        mm = A.dot(a - conv)
        self.embeddings = self.get_embedding_dense(mm, self.emb_size)
        return self.embeddings

    def transform(self):
        """
        Return final embedding dataframe.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [nodes, ProNE_Emb_0, ..., ProNE_Emb_N].
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not computed. Call `fit` and `chebyshev_gaussian` first.")

        emb_df = pd.DataFrame(self.embeddings)
        emb_df.columns = [f"ProNE_Emb_{i}" for i in range(emb_df.shape[1])]
        emb_df = emb_df.reset_index().rename(columns={"index": "nodes"})
        return emb_df.sort_values("nodes").reset_index(drop=True)
