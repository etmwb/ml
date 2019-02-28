import numpy as np
import cvxopt

class SVMTrainer(): 
	def __init__(self, kernel, c): 
		self._kernel = kernel
		self._c = c

	def train(self, X, y): 
		lagrange_multipliers = self._compute_multipliers(X, y)
		return self._construct_predictor(X, y, lagrange_multipliers)

	def _gram_matrix(self, X): 
		n_samples, n_features = X.shape
		K = np.zeros((n_samples, n_samples))
		for i in range(n_samples): 
			for j in range(n_samples): 
				K[i, j] = self._kernel(x[i], x[j])

		return K

	def _compute_multipliers(self, X, y): 
		n_samples, n_features = X.shape

		K = self._gram_matrix(X)

		# we will use cvxopt to solve quadratic programming, more information is available in 
		# https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

		P = cvxopt.matrix(np.outer(y, y) * K)
		q = cvxopt.matrix(-1 * np.ones(n_samples)) 

		# -\alpha_{i} \leq 0
		G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
		h_std = cvxopt.matrix(np.zeros(n_samples))

		# \alpha_{i} \leq C
		G_soft = cvxopt.matrix(np.diag(np.ones(n_samples)))
		h_soft = cvxopt.matrix(np.ones(n_samples) * self._c)

		G = cvxopt.matrix(np.vstack((G_std, G_soft)))
		h = cvxopt.matrix(np.vstack((h_std, h_soft)))

		A = cvxopt.matrix(y, (1, n_samples))
		b = cvxopt.matrix(0.0)

		solution = cvxopt.solvers.qp(P, q, G, h, A, b)

		return np.ravel(solution['x'])

	def _construct_predictor(self, X, y, lagrange_multipliers): 
		sv_indices = lagrange_multipliers > 0 
		sv_multiplier = lagrange_multipliers[sv_indices]
		sv = X[sv_indices]
		sv_y = y[sv_indices]

		# we calculate b with all sv rather than just one sv
		bias = np.mean(
			[y - SVMPredictor(self._kernel, 0.0, sv_multiplier, 
						sv, sv_y).forward(x) 
			for x, y in zip(sv, sv_y)])

		return SVMPredictor(self._kernel, bias, sv_multiplier, 
						sv, sv_y)


def SVMPredictor(): 
	def __init__(self, kernel, bias, alpha sv, sv_y): 
		self._kernel = kernel 
		self._bias = bias
		self._alpha = alpha
		self._sv = sv 
		self._sv_y = sv_y

	def forward(self, x): 
		result = self._bias
		for alpha_i, x_i, y_i in zip(alpha, sv, sv_y): 
			result += alpha_i * y_i * self._kernel(x_i, x)	

		return result