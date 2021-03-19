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

def test_metrics(offers, metrics, c_min=0, c_max=3, c_fid=4000):
	for m in metrics:
		for c_in, c in enumerate(np.linspace(c_min, c_max, c_fid)):
			m.test(offers, c, c_in=c_in)

def generate_auctions(num_offers, num_auctions, choices=[dist.Normal_Dist]):
	choices = [dist.Gamma_Dist, dist.Normal_Dist, dist.Uniform_Dist]
	return [[random.choice(choices)() for o in range(num_offers)] for a in range(num_auctions)]

if __name__ == '__main__':
	metrics = [
						#	metric.Dist_Selectivity_GivenTie(FID),
						#	metric.Dist_Selectivity_GivenTie_AllTies(FID),
							metric.Dist_Selectivity(FID),
							metric.Dist_Revenue_Impact(FID)
						]
	trials = {
						'normal': [dist.Normal_Dist],
						'uniform':[dist.Uniform_Dist],
						'gamma':  [dist.Gamma_Dist],
						'hightail':[dist.HighTail_Dist],
						#'gn':     [dist.Normal_Dist, dist.Gamma_Dist],
						#'gu':     [dist.Gamma_Dist, dist.Uniform_Dist],
						#'nu':     [dist.Normal_Dist, dist.Uniform_Dist],
						'all':    [dist.Normal_Dist, dist.Uniform_Dist, dist.Gamma_Dist, dist.HighTail_Dist]
					}

	for trial in trials:
		for i in range(4, 41, 4):
			auc = generate_auctions(i, FID, choices=trials[trial])
			for a in tqdm(auc, desc="{} offer auctions".format(i)):
				test_metrics(a, metrics, c_fid=FID)
		for m in metrics:
			m.graph(fname='images/{}_{}.png'.format(trial, m.name), show=False, save=True)
