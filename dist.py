import numpy as np
import random
import statistics
import abc
import math
from scipy import stats

UNIFORM_MIN_A = 0
UNIFORM_MAX_A = 10

class Dist:
	def __init__(self):
		self.est = self.generate_val()

	def get_offer_type(self):
		return OFFER_TYPES[self.offer_type]

	@abc.abstractmethod
	def generate_val(self):
		pass

	@abc.abstractmethod
	def adjusted(self, c):
		pass

	@abc.abstractmethod
	def mean(self):
		pass

	@abc.abstractmethod
	def variance(self):
		pass

class Normal_Dist(Dist):
	def __init__(self, stage_1_mean, stage_1_std, loc=None, std=None):
		if loc:
			self.loc = loc
		elif stage_1_mean:
			self.loc = stage_1_mean.generate_val()
		else:
			self.loc = random.randint(7, 10)

		if std:
			self.std = std
		elif stage_1_std:
			self.std = stage_1_std.generate_val()
		else:
			self.std = random.uniform(0, 3)
		super().__init__()

	def generate_val(self):
		return np.random.normal(loc=self.loc, scale=self.std)

	def adjusted(self, c):
		return self.est - c * self.std

	def mean(self):
		return self.loc

	def variance(self):
		return self.std**2

class Uniform_Dist(Dist):
	def __init__(self, stage_1_mean, stage_1_std, ab_pair=None, mean=None, min_a=0, max_a=10):
		if ab_pair ==None:
			self.m = stage_1_mean.generate_val() if stage_1_mean else mean
			self.std = stage_1_std.generate_val()
			self.a = self.m - math.sqrt(3)*self.std
			self.b = self.m + math.sqrt(3)*self.std
		else:
			self.a, self.b = ab_pair
			self.m= 0.5 * (self.a + self.b)
			self.std = math.sqrt(((self.b - self.a)**2)/12.0)
		super().__init__()

	def generate_val(self):
		return np.random.uniform(self.a, self.b)

	def adjusted(self, c):
		return self.est - c * self.std

	def mean(self):
		return self.m

	def variance(self):
		return self.std**2

DEFAULT_PHASE_1_MEAN = Normal_Dist(None, None, loc=10, std=2)
DEFAULT_PHASE_1_STD = Uniform_Dist(None, None, ab_pair=(0,3))

class Gamma_Dist(Dist):
	def __init__(self,  stage_1_mean, stage_1_std, mean=None, var=None):
		self.m = 0.1 + stage_1_mean.generate_val() if mean==None else mean
		self.var = 0.001 + stage_1_std.generate_val()**2 if var==None else var
		self.k = self.m**2 / self.var
		#self.theta = (self.m + self.var)/(self.m+self.k)
		self.theta = self.var / self.m
		super().__init__()

	def generate_val(self):
		return np.random.gamma(self.k, scale=self.theta)

	def adjusted(self, c):
		std_est = math.sqrt(self.k * self.theta**2)
		return self.est - c * std_est

	def mean(self):
		return self.k * self.theta

	def variance(self):
		return self.k * self.theta**2

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
		self.m = stage_1_mean.generate_val()+0.1 if mean==None else mean
		self.alpha = 2 + stage_1_std.generate_val()
		self.xm = (self.alpha* self.m - self.m)/self.alpha
		#self.var = #(self.m**2)/ (self.alpha *(self.alpha -2))
		self.var = (self.alpha * self.m**2)/ ((self.alpha-1)**2 *(self.alpha -2))
		super().__init__()

	def generate_val(self):
		d = (1+np.random.pareto(self.alpha)) * self.xm
		return d

	def adjusted(self, c):
		std_est = math.sqrt(self.var)
		return self.est - c * std_est

	def mean(self):
		return self.m

	def variance(self):
		return self.var

def proposed_hypothetical(c=0.3, lm=0.5, abv_means=[0,1], abv_stds=None):
	if not abv_stds:
		abv_stds = [2*lm,1*lm,0]
	random.shuffle(abv_stds)

	offers= []
	for f in range(3):
		lc = random.sample(abv_means, 1)
		sc = abv_stds.pop()
		offers.append( [np.random.normal(loc=lc,scale=sc), lc, sc] )

	# 'nearly tied' is informally defined
	th = c
	highest_xi = max(offers, key=lambda x:x[0])
	near_ties = []
	for o in offers:
		if (o[0]+c > highest_xi[0] and o[0] != highest_xi[0] ):
			near_ties.append(o)

	if len(near_ties) > 0:
		lower_std = min(near_ties, key= lambda x: x[2])
		if lower_std[1] == highest_xi[1]:
			return
		if lower_std[2] < highest_xi[2] and lower_std[1] > highest_xi[1]:
			return 1
		if lower_std[2] > highest_xi[2] and lower_std[1] < highest_xi[1]:
			return 1
		return 0

def est_p(c=0):
	acc = 0
	denom = 0
	while(denom < 10000):
		l = proposed_hypothetical(c=0.3)
		if l == None:
			continue
		denom += 1
		acc += l
		print(acc / denom, denom ,sep='\t',end='\r')
	return acc/denom

#est_p()


def dr_dc(offers):
	pairs = [(a, b) for idx, a in enumerate(means) for b in means[idx + 1:]]
	acc = 0
	for p in pairs:
		acc +=0
	return pairs

