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

from data import prepare_data, compute_rfa
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
import pickle
import seaborn as sns
sns.set()

def create_output_name(opt):
    titlename = f"dist={opt.distfn}, " +\
                f"metric={opt.distlocal}, " +\
                f"knn={opt.knn}, " +\
                f"loss={opt.lossfn} " +\
                f"sigma={opt.sigma:.2f}, " +\
                f"gamma={opt.gamma:.2f}, " +\
                f"n_pca={opt.pca}"

    output_dir = opt.dest + opt.family + '/'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    filename = output_dir +\
               f"PM{opt.knn:d}" +\
               f"sigma={opt.sigma:.2f}" +\
               f"gamma={opt.gamma:.2f}" +\
               f"{opt.distlocal}pca={opt.pca:d}_seed{opt.seed}"

    return titlename, filename



def get_tree_colors(opt, labels, tree_cl_name):
    pkl_file = open(f'{tree_cl_name}.pkl', 'rb')
    colors = pickle.load(pkl_file)
    colors_keys = [str(k) for k in colors.keys()]
    colors_val = [str(k) for k in colors.values()]
    colors = dict(zip(colors_keys, colors_val))
    pkl_file.close()
    tree_levels = []
    for l in labels:
        if l == '000':
            tree_levels.append('root')
        else:
            tree_levels.append(colors[l])

    tree_levels = np.array(tree_levels)
    n_tree_levels = len(np.unique(tree_levels))
    current_palette = sns.color_palette("husl", n_tree_levels)
    color_dict = dict(zip(np.unique(tree_levels), current_palette))
    sns.palplot(current_palette)
    color_dict[-1] = '#bdbdbd'
    color_dict['root'] = '#000000'
    return tree_levels, color_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Poincare maps')
    parser.add_argument('--dim', help='Embedding dimension', type=int, default=2)

    parser.add_argument('--path', help='Path to dataset to embed', type=str, default='../data_proteins/')
    parser.add_argument('--family', help='Name of the protein family (name of the folder)', type=str, default='glob')
    parser.add_argument('--tree', help='File with phylogenetic trees', type=str, default=5)
    parser.add_argument('--function', help='Protein by function', type=str, default='glob-name')

    parser.add_argument('--dest', help='Write results', type=str, default='../results_proteins/')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)

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

    parser.add_argument('--knn', help='Number of nearest neighbours in KNN', type=int, default=3)
    parser.add_argument('--connected', help='Force the knn graph to be connected', type=int, default=1)

    parser.add_argument('--sigma', help='Bandwidth in high dimensional space', type=float, default=1.0)
    parser.add_argument('--gamma', help='Bandwidth in low dimensional space', type=float, default=1.0)

    # optimization parameters
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.1)
    parser.add_argument('--lrm', help='Learning rate multiplier', type=float, default=1.0)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=5000)
    parser.add_argument('--batchsize', help='Batchsize', type=int, default=4)
    parser.add_argument('--burnin', help='Duration of burnin', type=int, default=500)

    parser.add_argument('--earlystop', help='Early stop  of training by epsilon. If 0, continue to max epochs', 
        type=float, default=0.0001)

    parser.add_argument('--debugplot', help='Plot intermidiate embeddings every N iterations', type=int, default=200)
    parser.add_argument('--cuda', help='Use GPU', type=int, default=1)
    parser.add_argument('--logfile', help='Use GPU', type=str, default='Logs')

    return parser.parse_args()

def poincare_map(opt):
    # read and preprocess the dataset
    features, labels = prepare_data(opt.path + opt.family + '/Nfasta/')

    if not (opt.tree is None):
        tree_levels, color_dict = get_tree_colors(opt, labels, f'{opt.path}/{opt.family}/{opt.family}_tree_cluster_{opt.tree}')
    else:
        color_dict = None
        tree_levels = None

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

    opt.lr = opt.batchsize / 16 * opt.lr

    titlename, fout = create_output_name(opt)

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
                                                     tree_levels=tree_levels,
                                                     earlystop=opt.earlystop,
                                                     color_dict=color_dict)


    df_pm = pd.DataFrame(embeddings, columns=['pm1', 'pm2'])
    df_pm['proteins'] = labels
    df_pm.to_csv(fout + '.csv', sep=',', index=False)


    t = timeit.default_timer() - t_start
    titlename = f"{opt.family}\nloss = {loss:.3e}\ntime = {t/60:.3f} min"


    plot_poincare_disc(embeddings, labels, 
                       title_name=titlename,
                       labels=tree_levels, 
                       coldict=color_dict, file_name=fout, d1=8.5, d2=8.0, bbox=(1.2, 1.), leg=False)

    idx_root = np.where(tree_levels == 'root')[0]
    poincare_coord_rot = poincare_translation(-embeddings[idx_root, :], embeddings)


    if not (opt.function is None):
        for f in ['glob-spec', 'glob-spec1', 'glob-name1', 'glob-name']:
            fun_levels, color_dict_fun = get_tree_colors(opt, labels, f'{opt.path}/{opt.family}/{f}')
            plot_poincare_disc(poincare_coord_rot, fun_levels, 
                               title_name=titlename,
                               labels=fun_levels, 
                               coldict=color_dict_fun, 
                               file_name=f'{fout}_rotate_{f}', 
                               d1=8.5, d2=8.0, bbox=(1., 1.), leg=True)

    else:
        color_dict_fun = None
        fun_levels = None


    
    for t in range(1, 6):
        tree_levels, color_dict = get_tree_colors(opt, labels, f'{opt.path}/{opt.family}/{opt.family}_tree_cluster_{t}')
        if len(np.unique(tree_levels)) < 25:
            leg = True
        else:
            leg = False
        plot_poincare_disc(poincare_coord_rot, labels, 
                           title_name=titlename,
                           labels=tree_levels, 
                           coldict=color_dict, file_name=f'{fout}_rotate_cut{t}', d1=8.5, d2=8.0, bbox=(1.2, 1.), leg=leg)


if __name__ == "__main__":
    args = parse_args()
    poincare_map(args)
