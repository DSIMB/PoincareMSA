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
import pandas as pd
from data import prepare_data, compute_rfa
from model import PoincareEmbedding, PoincareDistance
from model import poincare_root, poincare_translation
from rsgd import RiemannianSGD
from train import train
from poincare_maps import plotPoincareDisc
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

    if not os.path.isdir(opt.output_path):
        os.makedirs(opt.output_path)

    filename = opt.output_path +\
               f"PM{opt.knn:d}" +\
               f"sigma={opt.sigma:.2f}" +\
               f"gamma={opt.gamma:.2f}" +\
               f"{opt.distlocal}pca={opt.pca:d}_seed{opt.seed}"

    return titlename, filename



#def get_tree_colors(opt, labels, tree_cl_name):
#    pkl_file = open(f'{tree_cl_name}.pkl', 'rb')
#    colors = pickle.load(pkl_file)
#    colors_keys = [str(k) for k in colors.keys()]
#    colors_val = [str(k) for k in colors.values()]
#    colors = dict(zip(colors_keys, colors_val))
#    pkl_file.close()
#    tree_levels = []
#    for l in labels:
#        if l == 'root':
#            tree_levels.append('root')
#        else:
#            tree_levels.append(colors[l])
#
#    tree_levels = np.array(tree_levels)
#    n_tree_levels = len(np.unique(tree_levels))
#    current_palette = sns.color_palette("husl", n_tree_levels)
#    color_dict = dict(zip(np.unique(tree_levels), current_palette))
#    sns.palplot(current_palette)
#    color_dict[-1] = '#bdbdbd'
#    color_dict['root'] = '#000000'
#    return tree_levels, color_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Adaptation of Poincare maps for MSA')
    parser.add_argument('--dim', help='Embedding dimension', type=int, default=2)

    parser.add_argument('--input_path', help='Path to dataset to embed', type=str, 
        default='/Users/klanna/UniParis/PoincareMSA/data/glob/Nfasta/')

    parser.add_argument('--output_path', help='Path to dataset to embed', type=str, 
        default='/Users/klanna/UniParis/results/glob/')

    parser.add_argument('--plot',
        help='Flag True or False, if you want to plot the output.', type=str, 
        default='True')
    parser.add_argument('--checkout_freq',
        help='Checkout frequency (in epochs) to show intermidiate results', 
        type=int, default=10)

    parser.add_argument('--tree', 
        help='File with phylogenetic trees', type=str, default=5)
    parser.add_argument('--function', 
        help='Protein by function', type=str, default='glob-name')

    parser.add_argument('--seed',
        help='Random seed', type=int, default=0)

    parser.add_argument('--labels', help='has labels', type=int, default=1)
    parser.add_argument('--mode',
        help='Mode: features or KNN', type=str, default='features')

    parser.add_argument('--normalize',
        help='Apply z-transform to the data', type=int, default=0)
    parser.add_argument('--pca',
        help='Apply pca for data preprocessing (if pca=0, no pca)', 
        type=int, default=0)
    parser.add_argument('--distlocal', 
        help='Distance function (minkowski, cosine)', 
        type=str, default='cosine')
    parser.add_argument('--distfn', 
        help='Distance function (Euclidean, MFImixSym, MFI, MFIsym)', 
        type=str, default='MFIsym')
    parser.add_argument('--distr', 
        help='Target distribution (laplace, gaussian, student)', 
        type=str, default='laplace')
    parser.add_argument('--lossfn', help='Loss funstion (kl, klSym)',
        type=str, default='klSym')

#    parser.add_argument('--root', 
#        help='Get root node from labels', type=str, default="root")
    parser.add_argument('--iroot',
        help='Index of the root cell', type=int, default=0)
#    parser.add_argument('--rotate',
#        help='Rotate', type=int, default=-1)
    parser.add_argument('--rotate',
        help='use 0 element for calculations or not', action='store_true')

    parser.add_argument('--knn', 
        help='Number of nearest neighbours in KNN', type=int, default=5)
    parser.add_argument('--connected',
        help='Force the knn graph to be connected', type=int, default=1)

    parser.add_argument('--sigma',
        help='Bandwidth in high dimensional space', type=float, default=1.0)
    parser.add_argument('--gamma',
        help='Bandwidth in low dimensional space', type=float, default=2.0)

    # optimization parameters
    parser.add_argument('--lr',
        help='Learning rate', type=float, default=0.1)
    parser.add_argument('--lrm',
        help='Learning rate multiplier', type=float, default=1.0)
    parser.add_argument('--epochs',
        help='Number of epochs', type=int, default=1000)
    parser.add_argument('--batchsize',
        help='Batchsize', type=int, default=4)
    parser.add_argument('--burnin',
        help='Duration of burnin', type=int, default=500)

    parser.add_argument('--earlystop',
        help='Early stop  of training by epsilon. If 0, continue to max epochs', 
        type=float, default=0.0001)

    parser.add_argument('--debugplot',
        help='Plot intermidiate embeddings every N iterations',
        type=int, default=200)
    parser.add_argument('--logfile',
        help='Use GPU', type=str, default='Logs')
    args = parser.parse_args()

    args.plot = bool(args.plot)
    return args

def poincare_map(opt):
    # read and preprocess the dataset
    opt.cuda = True if torch.cuda.is_available() else False
    print('CUDA:', opt.cuda)
    torch.manual_seed(opt.seed)

    features, labels = prepare_data(opt.input_path, withroot = opt.rotate)
    print(len(features), len(labels)) 
    # if not (opt.tree is None):
    #     tree_levels, color_dict = get_tree_colors(
    #         opt, labels, 
    #         f'{opt.input_path}/{opt.family}_tree_cluster_{opt.tree}')
    # else:
    #     color_dict = None
    #     tree_levels = None

    # compute matrix of RFA similarities
    RFA = compute_rfa(
        features,
        mode=opt.mode,
        k_neighbours=opt.knn,
        distfn=opt.distfn,
        distlocal= opt.distlocal,
        connected=opt.connected,
        sigma=opt.sigma
        )
    print(RFA)
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
    predictor = PoincareEmbedding(
        len(dataset),
        opt.dim,
        dist=PoincareDistance,
        max_norm=1,
        Qdist=opt.distr, 
        lossfn = opt.lossfn,
        gamma=opt.gamma,
        cuda=opt.cuda
        )

    # instantiate the Riemannian optimizer 
    t_start = timeit.default_timer()
    optimizer = RiemannianSGD(predictor.parameters(), lr=opt.lr)

    # train predictor
    print('Starting training...')
    embeddings, loss, epoch = train(
        predictor,
        dataset,
        optimizer,
        opt,
        fout=fout,
        earlystop=opt.earlystop
        )

    df_pm = pd.DataFrame(embeddings, columns=['pm1', 'pm2'])
    df_pm['proteins_id'] = labels

    if opt.rotate:
        idx_root = np.where(df_pm['proteins_id'] == str(opt.iroot))[0][0]
        print("Recentering poincare disk at ", opt.iroot) 
#        print("root index: ", idx_root)
        poincare_coord_rot = poincare_translation(
            -embeddings[idx_root, :], embeddings)
        df_rot = df_pm.copy()
        df_rot['pm1'] = poincare_coord_rot[:, 0]
        df_rot['pm2'] = poincare_coord_rot[:, 1]
        df_rot.to_csv(fout + '.csv', sep=',', index=False)

    else:
        df_pm.to_csv(fout + '.csv', sep=',', index=False)

    t = timeit.default_timer() - t_start
    titlename = f"\nloss = {loss:.3e}\ntime = {t/60:.3f} min"
    print(titlename)

    plotPoincareDisc(
        embeddings, 
        title_name=titlename,
        file_name=fout, 
        d1=5.5, d2=5.0, 
        bbox=(1.2, 1.),
        leg=False
        )

    # idx_root = np.where(tree_levels == 'root')[0]
    # poincare_coord_rot = poincare_translation(-embeddings[idx_root, :], embeddings)


    # if not (opt.function is None):
    #     for f in ['glob_tree_cluster_1']:
    #         fun_levels, color_dict_fun = get_tree_colors(opt, labels, f'{opt.input_path}/{opt.family}/{f}')
    #         plotPoincareDisc(poincare_coord_rot, fun_levels, 
    #                            title_name=titlename,
    #                            labels=fun_levels, 
    #                            coldict=color_dict_fun, 
    #                            file_name=f'{fout}_rotate_{f}', 
    #                            d1=8.5, d2=8.0, bbox=(1., 1.), leg=False)

    # else:
    #     color_dict_fun = None
    #     fun_levels = None
    
    # for t in range(1, 6):
    #     tree_levels, color_dict = get_tree_colors(opt, labels, f'{opt.input_path}/{opt.family}/{opt.family}_tree_cluster_{t}')
    #     if len(np.unique(tree_levels)) < 25:
    #         leg = True
    #     else:
    #         leg = False
    #     plotPoincareDisc(poincare_coord_rot, labels, 
    #                        title_name=titlename,
    #                        labels=tree_levels, 
    #                        coldict=color_dict, file_name=f'{fout}_rotate_cut{t}', d1=8.5, d2=8.0, bbox=(1.2, 1.), leg=leg)


if __name__ == "__main__":
    args = parse_args()
    poincare_map(args)
