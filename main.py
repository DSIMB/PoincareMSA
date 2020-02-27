# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.metrics.pairwise import pairwise_distances
from torch.utils.data import TensorDataset
from sklearn.decomposition import PCA
import numpy as np
import argparse
import torch

from data import create_output_name, prepare_data, compute_rfa
from model import PoincareEmbedding, PoincareDistance
from model import poincare_root, poincare_translation
from rsgd import RiemannianSGD
from train import train
from visualize import *
from coldict import *

import os
import os.path
# from pathlib import Path


import timeit

if __name__ == "__main__":
		# parse arguments
		parser = argparse.ArgumentParser(description='Poincare maps')
		parser.add_argument('--dim', help='Embedding dimension', type=int, default=2)

		parser.add_argument('--path', help='Dataset to embed', type=str, default='../data/')
		parser.add_argument('--dset', help='Dataset to embed', type=str, default='Acetyltransf')
		parser.add_argument('--dest', help='Write results', type=str, default='../results/')

		parser.add_argument('--labels', help='has labels', type=int, default=1)
		parser.add_argument('--mode', help='Mode: features or KNN', type=str, default='features')

		parser.add_argument('--normalize', help='Apply z-transform to the data', type=int, default=0)
		parser.add_argument('--pca', help='Apply pca for data preprocessing (if pca=0, no pca)', type=int, default=0)

		parser.add_argument('--distlocal', help='Distance function (minkowski, cosine)', type=str, default='cosine')

		parser.add_argument('--distfn', help='Distance function (Euclidean, MFImixSym, MFI, MFIsym)', type=str, default='MFIsym')
		parser.add_argument('--distr', help='Target distribution (laplace, gaussian, student)', type=str, default='laplace')
		parser.add_argument('--lossfn', help='Loss funstion (kl, klSym)', type=str, default='klSym')

		parser.add_argument('--root', help='Get root node from labels', type=str, default="root")
		parser.add_argument('--iroot', help='Index of the root cell', type=int, default=-1)
		parser.add_argument('--rotate', help='Rotate', type=int, default=-1)

		parser.add_argument('--knn', help='Number of nearest neighbours in KNN', type=int, default=2)
		parser.add_argument('--connected', help='Force the knn graph to be connected', type=int, default=1)

		parser.add_argument('--sigma', help='Bandwidth in high dimensional space', type=float, default=1.0)
		parser.add_argument('--gamma', help='Bandwidth in low dimensional space', type=float, default=1.0)

		# optimization parameters
		parser.add_argument('--lr', help='Learning rate', type=float, default=0.1)
		parser.add_argument('--lrm', help='Learning rate multiplier', type=float, default=1.0)
		parser.add_argument('--epochs', help='Number of epochs', type=int, default=5000)
		parser.add_argument('--batchsize', help='Batchsize', type=int, default=4)
		parser.add_argument('--burnin', help='Duration of burnin', type=int, default=500)

		parser.add_argument('--seed', help='Duration of burnin', type=int, default=0)

		parser.add_argument('--earlystop', help='Early stop  of training by epsilon. If 0, continue to max epochs', 
			type=float, default=0.0001)

		parser.add_argument('--debugplot', help='Plot intermidiate embeddings every N iterations', type=int, default=200)
		
		parser.add_argument('--cuda', help='Use GPU', type=int, default=0)

		parser.add_argument('--logfile', help='Use GPU', type=str, default='Logs')

		parser.add_argument('--tb', help='Tensor board', type=float, default=0)

		opt = parser.parse_args()
				
		color_dict = None

		# read and preprocess the dataset
		features, labels = prepare_data(opt.path + opt.dset + '/')

		# compute matrix of RFA similarities
		RFA = compute_rfa(features, mode=opt.mode,
											k_neighbours=opt.knn,
											distfn=opt.distfn,
											distlocal= opt.distlocal,
											connected=opt.connected,
												sigma=opt.sigma)

		if opt.batchsize < 0:
			opt.batchsize = min(512, int(len(RFA)/10))
			print('batchsize = ', opt.batchsize)
			# if opt.dset == "Moignard2015":
			#     opt.batchsize = 1500
		opt.lr = opt.batchsize / 16 * opt.lr


		itlename, fout = create_output_name(opt)

			# PCA of RFA baseline
			# pca_baseline = PCA(n_components=2).fit_transform(RFA)
			# plot2D(pca_baseline.T,
			#        labels,
			#        fout + '_PCARFA',
			#        'PCA of RFA\n' + titlename)

		 
			# build the indexed RFA dataset 
		indices = torch.arange(len(RFA))
		if opt.cuda:
			indices = indices.cuda()
			RFA = RFA.cuda()

		dataset = TensorDataset(indices, RFA)

		# instantiate our Embedding predictor
		predictor = PoincareEmbedding(len(dataset),
																	opt.dim,
																	dist=PoincareDistance,
																	max_norm=1,
																	Qdist=opt.distr, 
																	lossfn = opt.lossfn,
																	gamma=opt.gamma,
																	cuda=opt.cuda)

		# instantiate the Riemannian optimizer 
		t_start = timeit.default_timer()
		optimizer = RiemannianSGD(predictor.parameters(), lr=opt.lr)

		# train predictor
		print('Starting training...')
		embeddings, loss, epoch = train(predictor,
														 dataset,
														 optimizer,
														 opt,
														 fout=fout,
														 labels=labels,
														 tb=opt.tb,
														 earlystop=opt.earlystop,
														 color_dict=color_dict)

		# np.savetxt(fout + '.csv', embeddings, delimiter=",")

		df_pm = pd.DataFrame(embeddings, columns=['pm1', 'pm2'])
		df_pm['proteins'] = labels
		df_pm.to_csv(fout + '.csv', sep=',', index=False)


		t = timeit.default_timer() - t_start
		titlename = f"loss = {loss:.3e}\ntime = {t/60:.3f} min"
		print(titlename)

		# log_file = f'results/{opt.logfile}.csv'		
		# df_stats = pd.DataFrame(np.array([[opt.dset, opt.pca, opt.knn, opt.sigma, opt.gamma, opt.distlocal, 
		# 	loss, int(t), int(t/60), opt.seed, opt.cuda, opt.earlystop, epoch]]), 
		# 	columns = ['dataset', 'pca', 'knn', 'sigma', 'gamma', 'distance',
		# 	'loss', 'time (sec)', 'time (min)', 'seed', 'cuda', 'earlystop', 'max epochs'])

		# if os.path.isfile(log_file):
		# 	df_logs = pd.read_csv(log_file)
		# 	df_stats = pd.concat([df_logs, df_stats])

		# df_stats.to_csv(f'results/{opt.logfile}.csv', index=False, sep=',')



		plot_poincare_disc(embeddings, labels,
											 labels=None,
											 coldict=None,
											 file_name=fout, d1=6.5, d2=6.0)
