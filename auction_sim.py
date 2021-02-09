import util
from offer import Offer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metric
from tqdm import tqdm

FID = 1000

def generate_graph(df):
	x_axis = np.linspace(0,3,FID)
	for col in df:
		plt.plot(x_axis,df[col], label=col)
	plt.legend(loc=1,ncol=2)
	plt.xlabel("Penalty Coefficient-c")
	plt.ylabel("Revenue Impact")
	plt.show()

def test_metrics(offers, metrics, c_min=0, c_max=3, c_fid=3000):
	for m in metrics:
		for c_in, c in enumerate(np.linspace(c_min, c_max, c_fid)):
			m.test(offers, c, c_in=c_in)

def generate_auctions(num_offers, num_auctions):
	return [[Offer() for o in range(num_offers)] for a in range(num_auctions)]

if __name__ == '__main__':
	metrics = [metric.Revenue_Impact(FID), metric.Selectivity(FID)]
	acc = {}
	for i in range(4, 41, 4):
		auc = generate_auctions(i, FID)
		for a_in, a in enumerate(auc):
			test_metrics(a, metrics, c_fid=FID)
			print ('num offers:', i, 'auction_num:', a_in, end='\r')
	for m in metrics:
		m.graph()
	#		acc[str(i)] += test_metric(a, util.revenue_impact, c_fid=FID)
	#	acc[str(i)] /= FID
	#df = pd.DataFrame(acc)
	#generate_graph(df)
