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

import timeit


if __name__ == "__main__":
		# parse arguments
		parser = argparse.ArgumentParser(description='Poincare maps')
		parser.add_argument('--dim', help='Embedding dimension', type=int, default=2)

		parser.add_argument('--path', help='Dataset to embed', type=str, default='../QUERY_PSI/')
		parser.add_argument('--maxlen', help='Max length', type=int, default=872)

		parser.add_argument('--dest', help='Write results', type=str, default='../results/')
		parser.add_argument('--dset', help='Write results', type=str, default='test1')

		parser.add_argument('--labels', help='has labels', type=int, default=1)
		parser.add_argument('--normalize', help='Apply z-transform to the data', type=int, default=0)
		parser.add_argument('--pca', help='Apply pca for data preprocessing (if pca=0, no pca)', type=int, default=0)

		parser.add_argument('--distfn', help='Distance function (Euclidean, MFImixSym, MFI, MFIsym)', type=str, default='MFIsym')
		parser.add_argument('--distr', help='Target distribution (laplace, gaussian, student)', type=str, default='laplace')
		parser.add_argument('--lossfn', help='Loss funstion (kl, klSym)', type=str, default='klSym')

		parser.add_argument('--root', help='Get root node from labels', type=str, default="root")
		parser.add_argument('--iroot', help='Index of the root cell', type=int, default=-1)
		parser.add_argument('--rotate', help='Rotate', type=int, default=-1)

		parser.add_argument('--knn', help='Number of nearest neighbours in KNN', type=int, default=15)
		parser.add_argument('--connected', help='Force the knn graph to be connected', type=int, default=1)

		parser.add_argument('--sigma', help='Bandwidth in high dimensional space', type=float, default=1.0)
		parser.add_argument('--gamma', help='Bandwidth in low dimensional space', type=float, default=2.0)

		# optimization parameters
		parser.add_argument('--lr', help='Learning rate', type=float, default=0.01)
		parser.add_argument('--lrm', help='Learning rate multiplier', type=float, default=1.0)
		parser.add_argument('--epochs', help='Number of epochs', type=int, default=10000)
		parser.add_argument('--batchsize', help='Batchsize', type=int, default=-1)
		parser.add_argument('--burnin', help='Duration of burnin', type=int, default=100)

		parser.add_argument('--debugplot', help='Plot intermidiate embeddings every N iterations', type=int, default=0)
		
		parser.add_argument('--cuda', help='Use GPU', type=int, default=0)

		parser.add_argument('--tb', help='Tensor board', type=float, default=0)

		opt = parser.parse_args()
				
		color_dict = None
		


			# read and preprocess the dataset
		features, labels = prepare_data(opt.path, opt.maxlen, n_pca=opt.pca)
		# features, labels = prepare_data(opt.path + opt.dset,
		# 																with_labels=opt.labels,
		# 																normalize=opt.normalize,
		# 																n_pca=opt.pca)

		

		# compute matrix of RFA similarities
		RFA = compute_rfa(features,
											k_neighbours=opt.knn,
											distfn=opt.distfn,
											connected=opt.connected,
											sigma=opt.sigma)

		# quit()

		if opt.batchsize < 0:
			opt.batchsize = min(2000, int(len(RFA)/8))
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
		start = timeit.default_timer()
		optimizer = RiemannianSGD(predictor.parameters(), lr=opt.lr)

		# train predictor
		embeddings, loss = train(predictor,
														 dataset,
														 optimizer,
														 opt,
														 fout=fout,
														 labels=labels,
														 tb=opt.tb)

		np.savetxt(fout + '.csv', embeddings, delimiter=",")

		titlename = f"loss = {loss:.3e}\ntime = {(timeit.default_timer() - start)/60:.3f} min"

		plot_poincare_disc(embeddings,
													 labels=labels,
													 coldict=color_dict,
													 file_name=fout)


		# rotation
		# root_hat = poincare_root(opt, labels, features)   
		# if root_hat != -1:
		# 		titlename = '{0}\nloss = {1:.3e} rotated'.format(titlename, loss)

		# 		poincare_coord_new = poincare_translation(
		# 				-embeddings[root_hat, :], embeddings)

		# 		plot_poincare_disc(poincare_coord_new,
		# 											 labels=labels,
		# 											 coldict=color_dict,
		# 											 file_name=fout + '_rotated')
