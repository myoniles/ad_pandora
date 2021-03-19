import numpy as np
import random
import statistics
import abc
import math

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
	def __init__(self, loc=None, std=None):
		self.loc = random.randint(7, 10) if loc == None else loc
		self.std = random.uniform(0, 3) if std == None else std
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
	def __init__(self, ab_pair=None, mean=None):
		if ab_pair ==None:
			if mean == None:
				self.a = random.uniform(UNIFORM_MIN_A,UNIFORM_MAX_A)
				self.b = random.uniform(self.a, 2*UNIFORM_MAX_A)
				self.m= 0.5 * (self.a + self.b)
			else:
				self.m= mean
				self.a = random.uniform(UNIFORM_MIN_A,UNIFORM_MAX_A)
				self.b = 2*mean - self.a
			self.var = ((self.b - self.a)**2)/12.0
		else:
			self.a, self.b = ab_pair
			self.m= 0.5 * (self.a + self.b)
			self.var = ((self.b - self.a)**2)/12.0
		super().__init__()

	def generate_val(self):
		return np.random.uniform(self.a, self.b)

	def adjusted(self, c):
		return self.est - c * math.sqrt(self.var)

	def mean(self):
		return self.m

	def variance(self):
		return self.var


DEFAULT_PHASE_1_MEAN = Normal_Dist(loc=10, std=2)
DEFAULT_PHASE_1_STD = Uniform_Dist(ab_pair=(0,3))

class Gamma_Dist(Dist):
	def __init__(self, mean=None, var=None):
		self.m = DEFAULT_PHASE_1_MEAN.generate_val() if mean==None else mean
		self.var = DEFAULT_PHASE_1_STD.generate_val() if var==None else var
		self.k = random.uniform(0,10)
		self.theta = (self.m + self.var)/(self.m+self.k)
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

print('\n')

def dr_dc(offers):
	pairs = [(a, b) for idx, a in enumerate(means) for b in means[idx + 1:]]
	acc = 0
	for p in pairs:
		acc +=0
	return pairs

k = Gamma_Dist(var=3)
print(k.mean(), k.variance(), k.k, k.theta)

#print(dr_dc([0,1]))

#print(acc / denom)

# Runnign this for simply 2 does nto work
# gives a 50% chance

# Running this with three yields about 0.7
