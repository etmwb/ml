import numpy as np 

class Kernel(): 
	@staticmethod 
	def linear(): 
		def f(x, y): 
			return np.inner(x, y)
		return f

	@staticmethod 
	def gaussian(sigma): 
		def f(x, y): 
			exponent = -np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
			return np.exp(exponent)
		return f 

	@staticmethod
	def poly(dim, off): 
		def f(x, y): 
			return (off + np.dot(x, y)) ** dim
		return f