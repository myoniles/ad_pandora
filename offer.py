import abc
import math
import numpy as np
import random

OFFER_TYPES= {0:"CPM", 1:"CPC", 2:"CPA"}
CPC_PSTAR_VALS = [0.005, 0.01, 0.02, 0.05]
CPA_PSTAR_VALS = [0.0002, 0.0005, 0.0001, 0.002]
N_VALS = [5000, 10000, 50000, 100000]

# mean of 1, same as normal distribution
GAMMA_PAIRS = [(1,1), (0.5, 2), (2, 0.5), (1/3, 3), (3, 1/3), (0.75, 4/3), (4/3, 0.75)]

class Offer:
	def __init__(self, cpc=CPC_PSTAR_VALS, cpa=CPA_PSTAR_VALS, n=N_VALS):
		self.offer_type = random.randint(0,2)
		if self.offer_type == 1:
			self.act_rate = random.choice(cpc)
		elif self.offer_type == 2:
			self.act_rate = random.choice(cpa)
		else:
			self.act_rate = 1.0 #CPM

		self.n = None
		self.est_rate = None
		self.act_val = None
		self.bid = None
		self.act_offer_val = None
		self.est_offer_val = None

		self.generate_val()
		self.generate_self(n)

	def generate_self(self, nvals):
		self.n = random.choice(nvals)
		self.est_rate = np.random.binomial(self.n, self.act_rate) / self.n
		self.bid = self.act_val / self.act_rate

		self.act_offer_val = self.bid * self.act_rate
		self.est_offer_val = self.bid * self.est_rate

	def get_offer_type(self):
		return OFFER_TYPES[self.offer_type]

	@abc.abstractmethod
	def generate_val(self):
		pass

	@abc.abstractmethod
	def adjusted_probability_estimate(self, c):
		pass

class Normal_Offer(Offer):
	def __init__(self, loc=1, std=0.1):
		self.loc = loc
		self.std = std
		super().__init__()

	def generate_val(self):
		self.act_val = np.random.normal(loc=self.loc, scale=self.std)

	def adjusted_probability_estimate(self, c):
		std_est = math.sqrt(self.est_rate*(1-self.est_rate)/self.n)
		return self.est_rate - c * std_est

class Gamma_Offer(Offer):
	def __init__(self, gamma_pairs=GAMMA_PAIRS):
		self.k, self.theta = random.choice(gamma_pairs)
		super().__init__()

	def generate_val(self):
		self.act_val = np.random.gamma(self.k, scale=self.theta)

	def adjusted_probability_estimate(self, c):
		std_est = math.sqrt(self.k * self.theta**2)
		return self.est_rate - c * std_est

class Poisson_Offer(Offer):
	def __init__(self):
		self.l = 1 # It is boring but we need a mean 1?
		super().__init__()

	def generate_val(self):
		self.act_val = np.random.poisson(lam=self.l)

	def adjusted_probability_estimate(self, c):
		std_est = self.l
		return self.est_rate - c * std_est
