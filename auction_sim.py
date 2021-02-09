import util
from offer import Offer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metric
from tqdm import tqdm

FID = 1000

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
			print ( ' '*35, end='\r')
			print ('num offers:', i, 'auction_num:', a_in+1, end='\r')
	for m in metrics:
		m.graph()
