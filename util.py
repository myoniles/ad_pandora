import random
import numpy as np
import scipy.integrate as intg
import math

def second_price( products ):
	# finds the winning selection for a second price auction based on the product of bid and estimated respsonse rate
	# Expects a 1D numpy array
	# w = argmax product
	# s = argmax product (i != w)
	x = np.argpartition(products, -2)[-2:] # get top two index
	return x[1], x[0] # return w, s


# Both maximal expected revenue and expected revenue have (basically) the same calculation.
# I have separated them for readability

def expected_revenue(est_rates, bids, act_rates):
	# let expected revenue r be the revenue obtained based on estimates
	pb = np.multiply(est_rates, bids)
	w, s = second_price(pb)
	return est_rates[s] * bids[s] * act_rates[w] / est_rates[w]

def maximal_expected_revenue( act_rates, bids ):
	# Let maximal expected revenue r* be the revenue obtained by an auction based on actual response rates
	pb = np.multiply(act_rates, bids)
	w, s = second_price(pb)
	return act_rates[s] * bids[s] * act_rates[w] / act_rates[w]

def revenue_impact(est_rates, act_rates, bids):
	# Define revenue impact R to be the portion of maximal expected revenue foregone by using estimates
	r_star = maximal_expected_revenue(act_rates, bids)
	r = expected_revenue(est_rates, bids, act_rates)
	return (r_star - r) / r_star

def offers_split(offers, c):
	l = [(o.bid, o.act_rate, o.adjusted_probability_estimate(c)) for o in offers]
	return map(list, zip(*l))

def _norm_max_helper(x, n):
	return x/(math.sqrt(math.pi)) * (math.e**(x**2 / -2) * 2**(1/2 - n) * n *(math.erf(x/math.sqrt(2))+ 1)**(n-1))

max_exp_dict = {}

def get_expected_normal_max(n, loc=0, scale=1):
	if n in max_exp_dict:
		return loc + scale * max_exp_dict[n]
	else:
		max_exp_dict[n] = intg.quad(_norm_max_helper, -1 * np.inf, np.inf, args=(n))[0]
		return loc + scale * max_exp_dict[n]
