import math
import random
import numpy as np

OFFER_TYPES= {0:"CPM", 1:"CPC", 2:"CPA"}
CPC_PSTAR_VALS = [0.005, 0.01, 0.02, 0.05]
CPA_PSTAR_VALS = [0.0002, 0.0005, 0.0001, 0.002]
N_VALS = [5000, 10000, 50000, 100000]

class Offer:
	def __init__(self):
		self.offer_type = random.randint(0,2)
		if self.offer_type == 1:
			self.act_rate = random.choice(CPC_PSTAR_VALS)
		elif self.offer_type == 2:
			self.act_rate = random.choice(CPA_PSTAR_VALS)
		else:
			self.act_rate = 1.0 #CPM

		self.n = random.choice(N_VALS)
		self.est_rate = np.random.binomial(self.n, self.act_rate) / self.n
		self.act_val = np.random.normal(loc=1, scale=0.1)
		self.bid = self.act_val / self.act_rate

		self.act_offer_val = self.bid * self.act_rate
		self.est_offer_val = self.bid * self.est_rate

	def adjusted_probability_estimate(self, c):
		debug = self.est_rate*(1-self.est_rate)/self.n
		std_est = math.sqrt(debug)
		return self.est_rate - c * std_est

	def get_offer_type(self):
		return OFFER_TYPES[self.offer_type]