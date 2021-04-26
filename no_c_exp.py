import util
import offer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metric
import dist
import random
from tqdm import tqdm

FID = 1000
M_LOC = 10
M_STD = 2
stage_1_m = dist.Normal_Dist(None, None,  loc=M_LOC, std=M_STD)
stage_1_v = dist.Uniform_Dist(None, None, ab_pair=(0,3))

def test_metrics(offers, m ):
		e = m.test(offers, 0,  update=False)
		return e

def generate_auctions(num_offers, num_auctions, choices=[dist.Normal_Dist]):
	return [[random.choice(choices)(stage_1_m, stage_1_v) for o in range(num_offers)] for a in range(num_auctions)]

if __name__ == '__main__':
	metrics = [
							#metric.Dist_Selectivity_GivenTie(FID),
							#metric.Dist_Selectivity_GivenTie_AllTies(FID),
							#metric.Dist_Selectivity(FID),
							#	metric.Dist_Revenue_Impact(FID),
							#metric.Dist_Revenue_Performance(FID),
							metric.Dist_MeanDiff(FID, m_loc=M_LOC, m_scale=M_STD)
						]
	trials = {
						'normal': [dist.Normal_Dist],
						#'uniform':[dist.Uniform_Dist],
						#'gamma':  [dist.Gamma_Dist],
						#'hightail':[dist.HighTail_Dist],
						#'all':    [dist.Normal_Dist, dist.Uniform_Dist, dist.Gamma_Dist, dist.HighTail_Dist]
					}

	x = list(range(3000,9000, 500))#[5, 100, 1000, 5000, 10000]
	y = []
	for trial in trials:
		for i in x:
			acc = 0
			auc = generate_auctions(i, 500, choices=trials[trial])
			for a in tqdm(auc, desc="{} offer auctions".format(i)):
				acc += test_metrics(a, metric.Dist_MeanDiff(FID))
			acc /= len(auc)
			print(acc)
			y.append(acc)
		for m in metrics:
			plt.plot(x,y,'r')
			plt.show()
