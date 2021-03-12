import sys
sys.path.insert(0, "/Users/klanna/pretty-plots")

import numpy as np
# import pplots
from pplots import *
import pandas as pd
import pickle

class PoincareMSA:
    """
    API for PoincareMSA for easy running and plotting.
    """
    def __init__(
        self,
        emb_file,
        save_folder='results/glob/'
    ):
        """
        Parameters
        ----------
        emb_file: str
            Full path to the file with pre-computed embedding.
        save_folder: optional (defaul: 'results/glob/')
            Full path to the folder where to store the results.
        """    
        embedding = pd.read_csv(emb_file)
        self.emb = embedding[['pm1', 'pm2']].values
        self.emb_file = emb_file
        self.save_folder = save_folder
        model_prefix = emb_file.split('/')[-1].split('.csv')[0]
        self.prefix = f"{self.save_folder}/{model_prefix}"
        self.labels = {}
        self.labels_palettes = {}
        self.n_proteins = len(self.emb)

        if embedding.columns[-1] != 'pm2':
            self.labels['proteins'] = embedding[embedding.columns[-1]].values
            print('Found labels: ', embedding.columns[-1])
            self.labels_palettes['proteins'] = dict(zip(self.labels['proteins'], ['#636363']*self.n_proteins)) # assign grey color
        else:
            print('WARNING! Protein labels are not found! It can lead to errors\
             in downstream functions.')

    def get_tree_colors(
        self, 
        colors_file, 
        colors_name,
        palette=None, 
        palette_name=None): 
        """
        Reads categories to color the map from a pickle file.
        Parameters
        ----------
        colors_file: str
            Full path to the pickle file.
        colors_name: str 
            Name of this category to save in this class.
        """  
        with open(colors_file, 'rb') as f:
            colors = pickle.load(f)
        colors['000'] = 'root'

        tree_levels = [colors[protein_id] for protein_id in pMSA.labels['proteins']]
        self.labels[colors_name] = np.array(tree_levels)
        self.labels_palettes[colors_name] = get_colors(list(np.unique(tree_levels)),\
            palette=palette, 
            palette_name=palette_name)

        # self.labels_palettes[-1] = '#bdbdbd'
        self.labels_palettes[colors_name][-1] = '#000000'
        self.labels_palettes[colors_name]['root'] = '#000000'

    def plot_dics(
        self, 
        labels_name=None, 
        file=None,
        postfix='',
        show_text=False,
        plot_legend=True,
        figsize=(7, 7)):

        if labels_name is None:
            labels = None
            col_dict = None
        else:
            if labels_name in self.labels.keys():
                labels = self.labels[labels_name]
                col_dict = self.labels_palettes[labels_name]
            else:
                print("This category name doesn't exist.")
                labels = None
                col_dict = None

        plot_embedding(
        self.emb, 
        labels,
        fig_width=figsize[0],
        fig_height=figsize[0],
        col_dict=col_dict,
        show_lines=False,
        show_text=show_text,
        title='My glob example',
        circe_transparency=0.5,
        file_name=f"{self.prefix}_{labels_name}{postfix}.png",
        is_hyperbolic=True,
        plot_legend=plot_legend
        )


if __name__ == "__main__":
    file_name = 'results/glob/PM3sigma=1.00gamma=1.00cosinepca=0_seed0.csv'

    pMSA = PoincareMSA(file_name)
    pMSA.get_tree_colors(
        'data/glob/glob_tree_cluster_1.pkl', 
        colors_name='glob_tree_cluster_1',        
        )

    # pMSA.plot_dics('proteins')
    pMSA.plot_dics('glob_tree_cluster_1',
        figsize=(12, 12),
        plot_legend=True)


    
