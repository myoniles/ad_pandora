import numpy as np
import random
import statistics
import abc
import math
from scipy import stats

UNIFORM_MIN_A = 0
UNIFORM_MAX_A = 10

class Dist:
	def __init__(self, m, s):
		self.m = m
		self.std = s
		self.est = max(0, self.generate_val())

	@abc.abstractmethod
	def generate_val(self):
		pass

	def adjusted(self, c):
		return self.est - c * self.std
		pass

	def mean(self):
		return self.m

	def variance(self):
		return self.std**2

class Normal_Dist(Dist):
	def __init__(self, stage_1_mean, stage_1_std, loc=None, std=None):
		l = stage_1_mean.generate_val() if loc==None else loc
		s = stage_1_std.generate_val() if std==None else std
		super().__init__(l,s)

	def generate_val(self):
		return np.random.normal(loc=self.m, scale=self.std)

class Uniform_Dist(Dist):
	def __init__(self, stage_1_mean, stage_1_std, ab_pair=None, mean=None, min_a=0, max_a=10):
		if ab_pair ==None:
			m = stage_1_mean.generate_val() if stage_1_mean else mean
			std = stage_1_std.generate_val()
			self.a = m - math.sqrt(3)*std
			self.b = m + math.sqrt(3)*std
		else:
			self.a, self.b = ab_pair
			m= 0.5 * (self.a + self.b)
			std = math.sqrt(((self.b - self.a)**2)/12.0)
		super().__init__(m, std)

	def generate_val(self):
		return np.random.uniform(self.a, self.b)


DEFAULT_PHASE_1_MEAN = Normal_Dist(None, None, loc=10, std=2)
DEFAULT_PHASE_1_STD = Uniform_Dist(None, None, ab_pair=(0,3))

class Gamma_Dist(Dist):
	def __init__(self,  stage_1_mean, stage_1_std, mean=None, var=None):
		m = 0.1 + stage_1_mean.generate_val() if mean==None else mean
		std = 0.001 + stage_1_std.generate_val()
		var = std**2 if var==None else var

		self.k = m**2 / var
		self.theta = var / m
		super().__init__(m, std )

	def generate_val(self):
		return np.random.gamma(self.k, scale=self.theta)

class HighTail_RandVar(stats.rv_continuous):
	def __init__(self, xm, alpha=5, xtol=1e-14, seed=None, stage_1_std=DEFAULT_PHASE_1_STD):
			self.xm = xm
			# If we want to keep variance 9 or below, we have to keep it greater than 3.18
			# I find this delightful since it is so close to pi
			self.alpha = 7.2 + stage_1_std.generate_val()
			super().__init__(a=0, xtol=xtol, seed=seed)

	def _cdf(self, x):
			return 1 - (self.xm / x)**self.alpha

class HighTail_Dist(Dist):
	def __init__(self, stage_1_mean, stage_1_std, mean=None, var=None):
		m = stage_1_mean.generate_val()+0.1 if mean==None else mean
		std = stage_1_std.generate_val()
		var = std**2

		self.alpha = math.sqrt((m**2+var)/var)+1
		assert(self.alpha>2)
		self.xm = m * (self.alpha-1) /self.alpha
		super().__init__(m, std)

	def generate_val(self):
		d = (1+np.random.pareto(self.alpha)) * self.xm
		return d
