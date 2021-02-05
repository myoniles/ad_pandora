import random
import numpy

def second_price( products ):
	# finds the winning selection for a second price auction based on the product of bid and estimated respsonse rate
	# Expects a 1D numpy array
	# w = argmax product
	# s = argmax product (i != w)
	x = np.argpartition(products, -2)[-2:] # get top two index
	return x[1], x[0] # return w, s


# Both maximal expected revenue and expected revenue have (basically) the same calculation.
# I have separated them for readability
# TODO change max_exp_rev to call exp_rev with act_rates for both rate args?

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
