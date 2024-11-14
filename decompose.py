# Based on code from https://github.com/AI4LIFE-GROUP/SpLiCE/tree/main
import torch
import torch.nn as nn
from sklearn import linear_model
from tqdm import trange, tqdm
import numpy as np
from sklearn.decomposition import SparseCoder


class ADMM:
    def __init__(
        self,
        rho=1.0,
        l1_penalty=0.2,
        tol=1e-6,
        max_iter=10000,
        device="cuda",
        verbose=False,
    ):
        self.rho = rho
        self.l1_penalty = l1_penalty
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        self.verbose = verbose

    def step(self, Cb, Q_cho, z, u):
        xn = torch.cholesky_solve(2 * Cb + self.rho * (z - u), Q_cho)
        zn = torch.where(
            (xn + u - self.l1_penalty / self.rho) > 0,
            xn + u - self.l1_penalty / self.rho,
            0,
        )
        un = u + xn - zn
        return xn, zn, un

    def fit(self, C, v):
        ## iterates are in c, the number of concepts
        c = C.shape[0]

        ## size: c x c
        Q = 2 * C @ C.T + (torch.eye(c) * self.rho).to(self.device)

        # factor Q for quicker solve -- this is critical.
        Q_cho = torch.linalg.cholesky(Q)

        # iterates, size: c x batch
        x = torch.randn((c, v.shape[0])).to(self.device)
        z = torch.randn((c, v.shape[0])).to(self.device)
        u = torch.randn((c, v.shape[0])).to(self.device)

        for ix in range(self.max_iter):
            z_old = z
            x, z, u = self.step(C @ v.T, Q_cho, z, u)

            res_prim = torch.linalg.norm(x - z, dim=0)
            res_dual = torch.linalg.norm(self.rho * (z - z_old), dim=0)

            if (res_prim.max() < self.tol) and (res_dual.max() < self.tol):
                break
        if self.verbose:
            print("Stopping at iteration {}".format(ix))
            print("Prime Residual, r_k: {}".format(res_prim.mean()))
            print("Dual Residual, s_k: {}".format(res_dual.mean()))
        return z.T


class Decompose(object):
    def __init__(
        self,
        vocab,
        l1_penalty: float = 0.15,
        solver="omp",
        transform_n_nonzero_coefs: float = 50,
    ):
        self.dictionary = vocab
        self.l1_penalty = l1_penalty
        self.solver = solver
        dim = vocab.shape[-1]
        if self.solver == "skl":
            self.l1_penalty = l1_penalty / (
                2 * dim
            )  ## skl regularization is off by a factor of 2 times the dimensionality of the CLIP embedding. See SKL docs.
        if self.solver == "admm":
            self.rho = 5
            self.tol = 1e-6
            self.admm = ADMM(
                rho=self.rho,
                l1_penalty=self.l1_penalty,
                tol=self.tol,
                max_iter=2000,
                device="cuda",
                verbose=False,
            )
        elif self.solver == "omp":
            self.coder = SparseCoder(
                transform_algorithm="omp",
                dictionary=vocab,
                transform_alpha=1e-4,
                transform_n_nonzero_coefs=transform_n_nonzero_coefs,
            )
        else:
            raise ValueError("Unknown decomposer")

    def transform(self, embedding, batch_size: int = 1024):
        """decompose Decomposes a dense CLIP embedding into a sparse weight vector

        Parameters
        ----------
        embedding : np.array
            A {batch x CLIP dimensionality} vector or batch of vectors.

        Returns
        -------
        weights : torch.tensor
            A {batch x num_concepts} sparse vector over concepts.
        """
        if self.solver == "skl":
            clf = linear_model.Lasso(
                alpha=self.l1_penalty,
                fit_intercept=False,
                positive=True,
                max_iter=10000,
                tol=1e-6,
            )
            skl_weights = []
            for i in trange(embedding.shape[0]):
                clf.fit(self.dictionary.T, embedding[i, :])
                skl_weights.append(torch.tensor(clf.coef_))
            weights = np.stack(skl_weights, axis=0)
        elif self.solver == "admm":
            weights = self.admm.fit(self.dictionary, embedding)
        elif self.solver == "omp":
            weights = []
            for i in trange(0, embedding.shape[0], batch_size):
                weights.append(self.coder.transform(embedding[i:i+batch_size]))
            weights = np.concatenate(weights, axis=0)
        else:
            raise ValueError("Unknown decomposer")
        return weights
