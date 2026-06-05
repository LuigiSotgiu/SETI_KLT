import numpy as np
from scipy import linalg

class MRKLT():
# Initialization:
    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None
        self.signal = None
        self.noise = None
        self.covariance_matrix = None
        self.reconstructed_signal = None

#=====================================================================================================================#
# Setters:
    def load_signal(self, X):
        self.signal = X

    def load_noise(self, noise):
        self.noise = noise

#=====================================================================================================================#
# Getters:
    def get_signal(self):
        return self.signal

    def get_noise(self):
        return self.noise

    def get_covariance_matrix(self):
        return self.covariance_matrix

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_eigenvectors(self):
        return self.eigenvectors

    def get_reconstructed_signal(self):
        return self.reconstructed_signal

#=====================================================================================================================#
    # Methods:
    def _compute_covariance_matrix(self):
        if self.noise is not None:
            self.covariance_matrix = np.cov(self.signal + self.noise, rowvar=False)
        else:
            if self.signal is not None:
                self.covariance_matrix = np.cov(self.signal, rowvar=False)
            else:
                print("Please load the signal before computing the covariance matrix.")
                self.covariance_matrix = None

    def _compute_eigenvalues_and_eigenvectors(self):
        if self.covariance_matrix is not None:
            self.eigenvalues, self.eigenvectors = linalg.eigh(self.covariance_matrix)
        else:
            print("Please compute the covariance matrix before computing eigenvalues and eigenvectors.")
            self.eigenvalues = None
            self.eigenvectors = None

    def _sort_eigenvalues_and_eigenvectors(self):
        if self.eigenvalues is not None and self.eigenvectors is not None:
            idx = np.argsort(self.eigenvalues)[::-1]
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:, idx]
        else:
            print("Please compute eigenvalues and eigenvectors before sorting.")

    def _compute_lambda(self):
        pass

#=====================================================================================================================#
    # Main method:
    def reconstruct_signal(self, n_eigenvectors=None):
        self._compute_covariance_matrix()
        self._compute_eigenvalues_and_eigenvectors()
        mu = np.mean(self.signal, axis=0)
        
        if n_eigenvectors is not None:
            self._sort_eigenvalues_and_eigenvectors()
            V = self.eigenvectors[:, :n_eigenvectors]
        else:
            V = self.eigenvectors

        zeta = (self.signal - mu) @ V
        self.reconstructed_signal = zeta @ V.conj().T + mu
        return self.reconstructed_signal