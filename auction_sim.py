import util
from offer import Offer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_graph(df):
	for col in df:
		plt.plot(df[col], label=col)
	plt.legend(loc=2,ncol=2)
	plt.show()

def test_metric(offers, metric, c_min=0, c_max=3, auc_num=10000):
	acc = []
	l = [(o.bid, o.act_rate) for o in offers]
	bids, act_rates = map(list, zip(*l))
	for c in np.linspace(c_min, c_max, auc_num):
		adj_est_rates = [o.adjusted_probability_estimate(c) for o in offers]
		acc.append(metric(adj_est_rates, act_rates, bids))
	return acc

if __name__ == '__main__':
	acc = {}
	for i in range(4, 41, 4):
		offers= [ Offer() for o in range(i)]
		acc[str(i)] = test_metric(offers, util.revenue_impact)
	df = pd.DataFrame(acc)
	generate_graph(df)
