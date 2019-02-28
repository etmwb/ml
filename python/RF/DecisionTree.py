import math
import numpy as np 

def std_agg(num, square_sum, sum): 
	"""
	E[(X-mean)**2] = E[X**2] - E[X]**2
	"""
	return math.sqrt((square_sum/cnt) - (sum/cnt)**2)

class DecisionTree(): 
	def __init__(self, n_features, feature_indices, idxs, depth=10, min_leaf=5): 
		r"""
		Args: 
			n_features (int): number of features used for each tree.
			feature_indices (numpy.array): available feature for cur node.
			idxs (numpy.array): avaliable indices of training data for cur node. 
			depth (int): remaining depth.
			min_leaf (int): minimum number of training data required to cause further split.
		"""
		self._n_features, self._fi, self._idxs, self._depth, self._ml = \
					n_features, feature_indices, idxs, depth, min_leaf
		self.n = len(self._idxs)
		self.val = np.mean(y[self._idxs])
		self.score = float('inf')

	def train(self, X, y): 
		self.X, self.y = X, y

		for indice in self._fi: self._find_split(indice)
		if self.is_leaf: return 
		li = np.nonzero(x <= self.split)[0]
		ri = np.nonzero(x > self.split)[0]
		lfi = np.random.permutation(self.X.shape[1])[:self._n_features]
		rfi = np.random.permutation(self.X.shape[1])[:self._n_features]
		self.left = DecisionTree(self._n_features, lfi, li, self._depth-1, self._ml).train(self.X, self.y)
		self.right = DecisionTree(self._n_features, rfi, ri, self._depth-1, self._ml).train(self.X, self.y)

	def _find_split(self, indice): 
		x, y = self.X[self._idxs, indice], self.y[self._idxs]
		sort_idx = np.argsort(x)

		sort_x, sort_y = x[sort_idx], y[sort_idx]
		lnum, lss, ls = 0, 0, 0
		rnum, rss, rs = self.n, (sort_y**2).sum(), sort_y.sum()

		for i in range(0, self.n-self._ml-1): 
			x_i, y_i = sort_x[i], sort_y[i]
			if i < self._ml or x_i == sort_x[i+1]: 
				continue

			lnum += 1; rnum -= 1
			lss += y_i ** 2; rss -= y_i ** 2
			ls += y_i; rs -= y_i

			lstd = std_agg(lnum, lss, ls)
			rstd = std_agg(rnum, rss, rs)
			cur_score = lstd * lnum + rstd * rnum
			if cur_score < self.score: 
				self.f_idx, self.score, self.split = indice, cur_score, x_i

	@property
	def is_leaf(self):
		return self.self.score == float('inf') or self.depth == 0

	def forward(self, x): 
		return np.array([self.recur(x_i) for x_i in x])

	def recur(self, x): 
		if self.is_leaf: return self.val 
		return self.left.recur(x) if x[self.f_idx] <= self.split else self.right.recur(x)