import util
import offer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metric
from tqdm import tqdm

FID = 100

def test_metrics(offers, metrics, c_min=0, c_max=3, c_fid=3000):
	for m in metrics:
		for c_in, c in enumerate(np.linspace(c_min, c_max, c_fid)):
			m.test(offers, c, c_in=c_in)

def generate_auctions(num_offers, num_auctions):
	return [[offer.Poisson_Offer() for o in range(num_offers)] for a in range(num_auctions)]

if __name__ == '__main__':
	metrics = [metric.Revenue_Impact(FID), metric.Selectivity(FID)]
	acc = {}
	for i in range(4, 41, 4):
		auc = generate_auctions(i, FID)
		for a in tqdm(auc, desc="{} offer auctions".format(i)):
			test_metrics(a, metrics, c_fid=FID)
	for m in metrics:
		m.graph()
