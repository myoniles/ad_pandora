import abc
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Metric(metaclass=abc.ABCMeta):

	def __init__(self, name, fidelity):
		self.name = name
		self.fidelity = fidelity
		self.acc = {}
		self.df = pd.DataFrame()

	@abc.abstractmethod
	def test(self, offers, c, update=True, c_in=0):
		pass

	def graph(self):
		self.df = pd.DataFrame(self.acc)
		self.df /= self.fidelity
		x_axis = np.linspace(0, 3, self.fidelity)
		for col in self.df:
			plt.plot(x_axis,self.df[col], label=col)
		plt.legend(loc=1, ncol=2)
		plt.xlabel("Penalty Coefficient-c")
		plt.ylabel(self.name)
		plt.show()

class Revenue_Impact(Metric):
	def __init__(self, fidelity):
		super().__init__('Revenue Impact', fidelity)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)

		bids, act_rates, est_rates = util.offers_split(offers, c)
		pb = np.multiply(est_rates, bids)
		w, s = util.second_price(pb)
		r_star = act_rates[s] * bids[s] * act_rates[w] / act_rates[w]
		r= est_rates[s] * bids[s] * act_rates[w] / est_rates[w]
		measure = (r_star - r) / r_star

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure


class Selectivity(Metric):
	def __init__(self, fidelity):
		super().__init__('Selectivity', fidelity)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)

		bids, act_rates, est_rates = util.offers_split(offers, c)
		pb_est = np.multiply(est_rates, bids)
		w_est, s_est = util.second_price(pb_est)
		pb_act = np.multiply(act_rates, bids)
		w_act, s_act = util.second_price(pb_act)
		measure = int(w_est == w_act)
		if update:
			self.acc[len(offers)][c_in] += measure
		return measure
