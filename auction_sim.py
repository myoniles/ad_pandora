import util
from offer import Offer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FID = 1000

def generate_graph(df):
	x_axis = np.linspace(0,3,FID)
	for col in df:
		plt.plot(x_axis,df[col], label=col)
	plt.legend(loc=1,ncol=2)
	plt.xlabel("Penalty Coefficient-c")
	plt.ylabel("Revenue Impact")
	plt.show()

def test_metric(offers, metric, c_min=0, c_max=3, c_fid=3000):
	acc = []
	l = [(o.bid, o.act_rate) for o in offers]
	bids, act_rates = map(list, zip(*l))
	i = 0
	for c in np.linspace(c_min, c_max, c_fid):
		i += 1
		adj_est_rates = [o.adjusted_probability_estimate(c) for o in offers]
		acc.append(metric(adj_est_rates, act_rates, bids))
	return acc

def generate_auctions(num_offers, num_auctions):
	return [[Offer() for o in range(num_offers)] for a in range(num_auctions)]

if __name__ == '__main__':
	acc = {}
	for i in range(4, 41, 4):
		acc[str(i)] = np.zeros(FID)
		auc = generate_auctions(i, FID)
		for a in auc:
			acc[str(i)] += test_metric(a, util.revenue_impact, c_fid=FID)
		acc[str(i)] /= FID
	df = pd.DataFrame(acc)
	generate_graph(df)
