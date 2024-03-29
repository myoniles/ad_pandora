import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metric
import dist
import random
from tqdm import tqdm
from scipy.optimize import curve_fit
import pickle


FID = 100
LINSPACE = np.linspace(0,10, FID)
stage_1_m = dist.Normal_Dist(None, None,  loc=100, std=10)
stage_1_v = dist.Uniform_Dist(None, None, ab_pair=(0,3))

def get_c_points(offers, m):
	for c_in, c in enumerate(LINSPACE):
		t = m.test(offers, c, c_in=c_in)

def objective(x, a, b, c, d):
	return ( a* x**3 ) + ( b * x**2 ) + ( c * x**1 ) + ( d * x**0 )


def generate_auctions(num_offers, num_auctions, choices=[dist.Normal_Dist]):
	return [[random.choice(choices)(stage_1_m, stage_1_v) for o in range(num_offers)] for a in range(num_auctions)]

if __name__ == '__main__':
	trials = {
						'normal': [dist.Normal_Dist],
						#'uniform':[dist.Uniform_Dist],
						'gamma':  [dist.Gamma_Dist],
						'hightail':[dist.HighTail_Dist],
						#'all':    [dist.Normal_Dist, dist.Uniform_Dist, dist.Gamma_Dist, dist.HighTail_Dist]
					}
	m = metric.Dist_Revenue_Performance(FID)

	for trial in trials:
		for i in range(10,1511,100):
			auc = generate_auctions(i, FID, choices=trials[trial])
			for a in tqdm(auc, desc="{} offer auctions".format(i)):
				get_c_points(a, m)
		p = m.get_max_points(LINSPACE)
		x, y = zip(*p)
		popt, _ = curve_fit(objective, x, y)
		print(popt, _)
		a, b, c, d = popt
		plt.scatter(x,y)
		x_line = np.arange(min(x), max(x), 1)
		y_line = objective(x_line, a, b, c, d )
		plt.plot(x_line, y_line, '--')
		plt.show()
