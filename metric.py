import abc
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class Metric(metaclass=abc.ABCMeta):

	def __init__(self, name, fidelity, ax_thresh=None):
		self.name = name
		self.fidelity = fidelity
		self.fidelity_gran = None
		self.acc = {}
		self.df = pd.DataFrame()
		self.ax_thresh = ax_thresh

	@abc.abstractmethod
	def test(self, offers, c, update=True, c_in=0):
		pass

	def graph(self, fname='plot', show=True, save=False, linspace_bounds=[0,3]):
		self.df = pd.DataFrame(self.acc)
		if self.fidelity_gran == None:
			self.df /= self.fidelity
		else:
			self.df = self.df.div(pd.DataFrame(self.fidelity_gran))
		x_axis = np.linspace(linspace_bounds[0], linspace_bounds[1], self.fidelity)
		for col in self.df:
			plt.plot(x_axis,self.df[col], label=col)
		plt.legend(loc=1, ncol=1)
		plt.xlabel("Penalty Coefficient-c")
		if self.ax_thresh:
			plt.axhline(self.ax_thresh, c='r', ls=':')
		plt.ylabel(self.name)
		if show:
			plt.show()
		if save:
			plt.savefig(fname)
		plt.clf()

	def refresh(self):
		self.fidelity_gran = None
		self.acc = {}
		self.df = pd.DataFrame()

	def get_max_points(self, linspace):
		# return a point (n, c)
		# This point corresponds to the c value when the metric was maximized for a given n
		points = []
		for n in self.acc:
			c = linspace[self.acc[n].argmax()]
			points.append((n,c))
		return points

class Offer_Revenue_Impact(Metric):
	def __init__(self, fidelity):
		super().__init__('Offer Revenue Impact', fidelity)

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


class Offer_Selectivity(Metric):
	def __init__(self, fidelity):
		super().__init__('Offer Selectivity', fidelity)

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

class Dist_Revenue_Impact(Metric):
	# Proportion of forgone expected maximal revenue
	def __init__(self, fidelity):
		super().__init__('Distribution Revenue Impact', fidelity)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)

		# r star is maximal expected revenue, that is to asy with clairvoyance
		# expected revenue in this case is just the largest mean
		r_star = max(offers, key=lambda x: x.mean())
		# expected revenue r is the mean of the dist using the value c
		r= max(offers, key=lambda x: x.adjusted(c))
		measure = (r_star.mean() - r.mean()) / r_star.mean()

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure

class Dist_Revenue_Performance(Metric):
	def __init__(self, fidelity):
		super().__init__('Distribution Revenue Performance', fidelity)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)

		# r star is maximal expected revenue, that is to asy with clairvoyance
		# expected revenue in this case is just the largest mean
		r_star = max(offers, key=lambda x: x.mean())
		# expected revenue r is the mean of the dist using the value c
		r= max(offers, key=lambda x: x.adjusted(c))
		assert(r_star.mean() >= r.mean())
		measure = r.mean() / r_star.mean() # (r_star.mean() - r.mean()) / r_star.mean()

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure

from scipy.stats import norm, t
import scipy.integrate as intg

class Dist_Selectivity_strat2(Metric):
	def __init__(self, fidelity, m_loc=0, m_scale=1):
		self.m_loc = m_loc
		self.m_std = m_scale
		super().__init__('Distribution Revenue Performance', fidelity)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)

		mu_0 = util.get_expected_normal_max(len(offers), loc=self.m_loc, scale=self.m_std)
		# Get best as determined by s2
		def helper(x):
			t1 = norm.cdf(mu_0+c*x.est, loc=mu_0, scale=math.sqrt(x.variance()))
			t2 = norm.cdf(mu_0-c*x.est, loc=mu_0, scale=math.sqrt(x.variance()))
			#x.est * norm.cdf(12, loc=10, scale=2) - norm.cdf(mu_0-c*x.est, loc=mu_0, scale=math.sqrt(x.variance()))
			return x.est * (t1 - t2)
		s2 = max(offers, key=helper )

		# r star is maximal expected revenue, that is to asy with clairvoyance
		# expected revenue in this case is just the largest mean
		high_est = max(offers, key=lambda x: x.est)
		# expected revenue r is the mean of the dist using the value c
		#r= max(offers, key=lambda x: t.sf((abs(x.m-self.best_est)/x.std), df=2))
		#r= b.mean()
		#print(norm.cdf((r.m-self.best_est)/r.std))
		measure = s2.mean() > high_est.mean() # (r_star.mean() - r.mean()) / r_star.mean()

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure
class Dist_Revenue_Performance_strat2(Metric):
	def __init__(self, fidelity, m_loc=0, m_scale=1):
		self.m_loc = m_loc
		self.m_std = m_scale
		super().__init__('Distribution Revenue Performance', fidelity)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)

		mu_0 = util.get_expected_normal_max(len(offers), loc=self.m_loc, scale=self.m_std)
		# Get best as determined by s2
		def helper(x):
			t1 = norm.cdf(mu_0+c*x.est, loc=mu_0, scale=math.sqrt(x.variance()))
			t2 = norm.cdf(mu_0-c*x.est, loc=mu_0, scale=math.sqrt(x.variance()))
			#x.est * norm.cdf(12, loc=10, scale=2) - norm.cdf(mu_0-c*x.est, loc=mu_0, scale=math.sqrt(x.variance()))
			#if(c< 0.3):
				#print(round(c,3),  round(t1 -t2, 3), round(x.est*(t1-t2), 3), end='\r')
			return x.est * (t1 - t2)
		r = max(offers, key=helper )
		if c <= 0.03:
			r = max(offers, key=lambda x: x.est)

		# r star is maximal expected revenue, that is to asy with clairvoyance
		# expected revenue in this case is just the largest mean
		r_star = max(offers, key=lambda x: x.mean())
		# expected revenue r is the mean of the dist using the value c
		#r= max(offers, key=lambda x: t.sf((abs(x.m-self.best_est)/x.std), df=2))
		#r= b.mean()
		#print(norm.cdf((r.m-self.best_est)/r.std))
		measure = r.mean() / r_star.mean() # (r_star.mean() - r.mean()) / r_star.mean()

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure

class Dist_Selectivity(Metric):
	def __init__(self, fidelity):
		super().__init__('Distribution Selectivity', fidelity)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)

		est_cc = max(offers, key=lambda x: x.adjusted(c))
		o = util.get_expected_normal_max(len(offers), loc=10, scale=2)
		o_in = max(offers, key=lambda x: x.mean())
		o = min(o, o_in.mean())
		measure = int(est_cc.mean() >= o)

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure

class Dist_Favorability_GivenTie_AllTies(Metric):
	def __init__(self, fidelity):
		super().__init__('Selectivity given near tie, all ties', fidelity, ax_thresh = 0.5)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)
		if not self.fidelity_gran: # first with n offers
			self.fidelity_gran = {}
		if len(offers) not in self.fidelity_gran: # first with n offers
			self.fidelity_gran[len(offers)] = np.zeros(self.fidelity)

		highest_xi = max(offers, key=lambda x: x.est)
		near_ties = []
		for o in offers:
			if (highest_xi.adjusted(c) <= o.adjusted(c) and o.mean() != highest_xi.mean() ):
				near_ties.append(o)

		measure = 0
		self.fidelity_gran[len(offers)][c_in] += len(near_ties)
		for lower_std in near_ties:
			if lower_std.variance() < highest_xi.variance() and lower_std.mean() > highest_xi.mean():
				measure += 1
			elif lower_std.variance() > highest_xi.variance() and lower_std.mean() < highest_xi.mean():
				measure += 1

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure

class Dist_Favorability_GivenTie(Metric):
	def __init__(self, fidelity):
		self.ax_thresh = 0.5
		super().__init__('Favorable breaks given near tie', fidelity, ax_thresh = 0.5)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)
		if not self.fidelity_gran: # first with n offers
			self.fidelity_gran = {}
		if len(offers) not in self.fidelity_gran: # first with n offers
			self.fidelity_gran[len(offers)] = np.zeros(self.fidelity)

		highest_xi = max(offers, key=lambda x: x.est)
		near_ties = []
		for o in offers:
			if (highest_xi.adjusted(c) <= o.adjusted(c) and o.mean() != highest_xi.mean() ):
				near_ties.append(o)

		measure = 0
		if len(near_ties) > 0:
			self.fidelity_gran[len(offers)][c_in] += 1
			lower_std = max(near_ties, key=lambda x: x.adjusted(c))
			if lower_std.variance() < highest_xi.variance() and lower_std.mean() > highest_xi.mean():
				measure += 1
			elif lower_std.variance() > highest_xi.variance() and lower_std.mean() < highest_xi.mean():
				measure += 1

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure

class Dist_MeanDiff(Metric):
	def __init__(self, fidelity):
		super().__init__('Difference in Rewards', fidelity)

	def test(self, offers, c, update=True, c_in=0):
		if len(offers) not in self.acc: # first with n offers
			self.acc[len(offers)] = np.zeros(self.fidelity)

		#est_c0 = max(offers, key=lambda x: x.est)
		highest_off = offers[0]
		second_highest_off = offers[0]
		for x in offers:
			if x.est >= highest_off.est:
				second_highest_off = highest_off
				highest_off = x

		measure = highest_off.mean() - second_highest_off.mean()

		#offers.sort(key=lambda x:-1*x.est)
		#measure = offers[0].mean() - offers[1].mean()

		if update:
			self.acc[len(offers)][c_in] += measure
		return measure

