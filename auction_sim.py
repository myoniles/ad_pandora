import util
import offer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metric
import dist
import random
from tqdm import tqdm

FID = 100
LINSPACE_BOUNDS = [0, 3]
stage_1_m = dist.Normal_Dist(None, None,  loc=10, std=3)
stage_1_v = dist.Uniform_Dist(None, None, ab_pair=(0,3))

def test_metrics(offers, metrics, c_min=0, c_max=3, c_fid=1000):
	for m in metrics:
		for c_in, c in enumerate(np.linspace(c_min, c_max, c_fid)):
			m.test(offers, c, c_in=c_in)

def generate_auctions(num_offers, num_auctions, choices=[dist.Normal_Dist]):
	return [[random.choice(choices)(stage_1_m, stage_1_v) for o in range(num_offers)] for a in range(num_auctions)]

if __name__ == '__main__':
	metrics = [
							#metric.Dist_Favorability_GivenTie(FID),
							#metric.Dist_Favorability_GivenTie_AllTies(FID),
							#metric.Dist_Selectivity(FID),
							#	metric.Dist_Revenue_Impact(FID),
							metric.Dist_Revenue_Performance(FID),
							#metric.Dist_MeanDiff(FID),
							#metric.Dist_Revenue_Performance_strat2(FID, m_loc=M_LOC, m_scale=M_STD)
							#metric.Dist_Selectivity_strat2(FID, m_loc=M_LOC, m_scale=M_STD)
						]
	trials = {
						#'normal': [dist.Normal_Dist],
						#'uniform':[dist.Uniform_Dist],
						#'gamma':  [dist.Gamma_Dist],
						#'hightail':[dist.HighTail_Dist],
						'all':    [dist.Normal_Dist, dist.Uniform_Dist, dist.Gamma_Dist, dist.HighTail_Dist]
					}

	for trial in trials:
		for m in metrics:
			m.refresh()
		for i in [4,8,12, 16,20, 24, 28, 32, 36, 40]:
			auc = generate_auctions(i, FID, choices=trials[trial])
			for a in tqdm(auc, desc="{} offer auctions".format(i)):
			#for a in auc:
				test_metrics(a, metrics, c_fid=FID, c_min=LINSPACE_BOUNDS[0], c_max=LINSPACE_BOUNDS[1])
		for m in metrics:
			m.graph(fname='images/highvar_last_{}_{}.png'.format(trial, m.name), show=True, save=False, linspace_bounds=LINSPACE_BOUNDS)
