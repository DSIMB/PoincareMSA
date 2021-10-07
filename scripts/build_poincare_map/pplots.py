import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adjustText import adjust_text

import matplotlib
matplotlib.use('Agg')

def read_embeddings(path_annotation, path_embedding):
    df = pd.read_csv(path_annotation)
    df['proteins_id'] = list(df.index)
    df = df.set_index(['proteins_id'])
    embeddings = pd.read_csv(path_embedding)
    embeddings = embeddings.set_index(['proteins_id'])
    result = pd.concat([embeddings, df], axis=1, join="inner")
    return result


def get_palette(
        n_colors, 
        palette_name='category20'
    ):

    try:
        palette = sns.color_palette(palette_name)
    except:
        print('Palette not found. Using default palette tab10')
        palette = sns.color_palette('category20')
    while len(palette) < n_colors:
        palette += palette
    
    return palette


# default parameters for matlab
FONT_SIZE = 7
font = {'size': FONT_SIZE}
matplotlib.rc('font', **font)
matplotlib.rc('ytick', labelsize=FONT_SIZE)
matplotlib.rc('xtick', labelsize=FONT_SIZE)

from sklearn.metrics.pairwise import cosine_similarity

def save_to_file(fig, file_name, file_format):
    if file_format is None:
        if file_name.split(".")[-1] in ['png', 'pdf']:
            file_format = file_name.split(".")[-1]
            savename = file_name                
        else:
            file_format = 'pdf'
            savename = f'{file_name}.{file_format}'
        
        fig.savefig(savename, format=file_format)
        print(f"Saved file to: {savename}")


def plot_embedding(
        df,
        labels=None,
        labels_text=None,
        labels_idx=None,
        col_dict=None,
        title=None,
        show_lines=False,
        show_text=False,
        show_legend=True,
        axis_equal=True,
        circle_size=30,
        circe_transparency=1.0,
        line_transparency=0.8,
        line_width=0.8,
        fontsize=9,
        fig_width=10,
        fig_height=10,
        file_name=None,
        file_format=None,
        labels_name=None,
        width_ratios=[7, 1],
        bbox=(1.3, 0.7),
        plot_legend=False,
        is_hyperbolic=True
    ):
    
    # create data structure suitable for embedding
    
    if not (labels_name is None):
        labels = df[labels_name].values
    else:
        labels_name = 'labels'
        df[labels_name] = 'na'
    
    if labels_text == 'all':
        labels_text = labels

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
    ax = plt.subplot(gs[0])
    
    sns.despine(left=False, bottom=False, right=True)
    
    # if col_dict is None:
    #     unique_labels = np.unique(labels)
    #     col_dict = dict(zip(unique_labels, ['#00000']*len(unique_labels)))
    
    if (col_dict is None) and not (labels is None):
        col_dict = get_colors(labels)

    if is_hyperbolic:
        circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
        plt.gca().add_patch(circle)
        ax.plot(0, 0, 'x', c=(0, 0, 0), ms=2)
        ax.axis('off')


    sns.scatterplot(
        x="pm1", 
        y="pm2", 
        hue=labels_name, 
#       hue_order=labels_order,
        palette=col_dict,
        # palette='Set1',
        alpha=circe_transparency,                    
        edgecolor="none",
        s=circle_size,
        data=df, 
        ax=ax)
    
    # I remove seaborn default legend so it won't clash with the rest of the plot
    if plot_legend is False:
        try:
            ax.legend_.remove()
        except:
            pass
    else:
        plt.legend(numpoints=1, loc='center left',
               bbox_to_anchor=bbox, fontsize=fontsize)
    
    emb = df[['pm1', 'pm2']].values
    if show_lines:        
        for i in range(len(emb)):
            if col_dict is None:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=None
                )
            else:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=col_dict[labels[i]]
                )

    if show_text and (not (labels_text is None)):
        texts = []
        if labels_idx is None:
            labels_idx = list(range(len(labels_text)))
        for i in labels_idx:
            texts.append(
                ax.text(
                    emb[i, 0], 
                    emb[i, 1], 
                    labels_text[i], 
                    fontsize=fontsize
                )
            )
    
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='black', lw=0.1),
            ax=ax
        )    

    
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold")

    ax.set_xlabel('pm1', fontsize=fontsize)
    ax.set_ylabel('pm2', fontsize=fontsize)

    if axis_equal:
        ax.axis('equal')
        ax.axis('square')

    plt.tight_layout()

    if file_name:
        save_to_file(fig, file_name, file_format)        

    return plt


def get_colors(labels, palette=None, palette_name=None):
    n_colors = len(labels)
    if palette is None:
        palette = get_palette(n_colors, palette_name)    
    col_dict = dict(zip(labels, palette[:n_colors]))
    return col_dict


def plot_similarity(
        emb,
        labels=None,
        col_dict=None,
        fig_width=4,
        fig_height=4,
        cmap='coolwarm',
        fmt='png',
        fontsize=7,
        file_format=None,
        file_name=None
    ):

    # first we take construct similarity matrix
    # add another similarity
    similarity_matrix = cosine_similarity(emb)

    df = pd.DataFrame(
        similarity_matrix,
        columns=labels,
        index=labels,
    )

    if col_dict is None:
        col_dict = get_colors(labels)

    network_colors = pd.Series(df.columns, index=df.columns).map(col_dict)

    sns_plot = sns.clustermap(
        df,
        cmap=cmap, 
        center=0, 
        row_colors=network_colors,
        col_colors=network_colors,
        mask=False,
        metric='euclidean',
        figsize=(fig_height, fig_width),
        vmin=-1, vmax=1,
        fmt=file_format
    )

    sns_plot.ax_heatmap.xaxis.set_tick_params(labelsize=fontsize)
    sns_plot.ax_heatmap.yaxis.set_tick_params(labelsize=fontsize)
    sns_plot.ax_heatmap.axis('equal')
    sns_plot.cax.yaxis.set_tick_params(labelsize=fontsize)

    if file_name:
        save_to_file(sns_plot, file_name, file_format)