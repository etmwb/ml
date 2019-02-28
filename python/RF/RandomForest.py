import numpy as np 

class RandomForest(): 
	def __init__(self, n_trees, n_features, n_samples, depth=10, min_leaf=5): 
		r"""
		Args: 
			n_trees (int): number of trees. 
			n_features (int): number of features used for each tree.
			n_samples (int): number of training data used for each tree. 
			depth (int): depth of each tree.
			min_leaf (int): minimum number of training data required to cause further split.
		"""
		self._n_trees, self._n_features, self._n_samples, self._depth, self._min_leaf = \
										n_trees, n_features, n_samples, depth, min_leaf

	def train(self, X, y): 
		np.random.seed(12) 
		self.X, self.y = X, y
		self.trees = [self._create_tree() for i in range(self._n_trees)]

	def _create_tree(self): 
		indices = np.random.permutation(len(self.y))[:self._n_samples] 
		feature_indices = np.random.permutation(self.X.shape[1])[:self._n_features]
		return DecisionTree(self._n_features, feature_indices, idxs=np.array(range(self._n_samples)), 
							depth=self._depth, min_leaf=self._min_leaf).train(self.X[indices], self.y[indices]) 

	def forward(self, X): 
		return np.mean([tree.forward(x) for tree in self.trees], axis=0)

