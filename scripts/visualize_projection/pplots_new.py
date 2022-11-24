import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adjustText import adjust_text
from scripts.visualize_projection.model import poincare_translation

import plotly.graph_objs as go
#import plotly.io as pio
import plotly
import plotly.express as px

import matplotlib
matplotlib.use('Agg')

import plotly.graph_objs as go

def rotate(df, root_name='root'):
    idx_root = np.where(df.index == root_name)[0][0]
    embeddings = df[['pm1', 'pm2']].values
    poincare_coord_rot = poincare_translation(
        -embeddings[idx_root, :], embeddings)
    df_rot = df.copy()
    df_rot['pm1'] = poincare_coord_rot[:, 0]
    df_rot['pm2'] = poincare_coord_rot[:, 1]
    return df_rot

"""
def read_embeddings(
    path_embedding, 
    path_annotation = None, 
    withroot = True
    #    root_name='root'
    ):

    embeddings = pd.read_csv(path_embedding)
    embeddings = embeddings.set_index(['proteins_id'])
    embeddings.index = embeddings.index.astype("int64") # convert all indices to numbers
    embeddings.sort_index(axis = 0, inplace = True)
#    embeddings.loc[root_name] = [root_name]*len(embeddings.columns)        

    if not (path_annotation is None):
        df = pd.read_csv(path_annotation)   
#        if withroot:
        df['proteins_id'] = [str(i+1) for i in list(df.index)]
#        else:
#            df['proteins_id'] = [str(i) for i in list(df.index)]    
        df = df.set_index(['proteins_id'])
        if withroot:
            df.loc[0] = "root"
            embeddings.loc[0] = float(0) 
            list_var_float = ['evo_distance', 'decile_domain']
            for var in list_var_float:
                if var in df.columns: 
                    df.loc[0, var] = float(0)
                    df[var] = df[var].astype('float64')  

        df.index = df.index.astype("int64")
        df.sort_index(axis = 0, inplace = True)
 #       df.loc[root_name] = [root_name]*len(df.columns)
        result = pd.concat([embeddings, df], axis=1, join="outer")
        list_var_object = ["tree1", "tree2", "tree3", "tree4", "Color_eukaryota"]
        for var in list_var_object:
            if var in result.columns:
                result[var] = result[var].astype("object")
        return result
    else:
        return embeddings
"""


def read_embeddings(path_embedding, path_annotation=None, withroot=True):
    """
    Prepare the plot data by concatenating the two dataframes.
    """
    df_embeddings = pd.read_csv(path_embedding)
    df_embeddings.sort_values(["proteins_id"], ignore_index=True, ascending=True, inplace=True)

    #If an annotation file is provided
    if os.path.isfile(path_annotation):
        df_annotation = pd.read_csv(path_annotation)
        #Check that no column is named "proteins_id" (already used in df_embeddings)
        if "proteins_id" in df_annotation.columns:
            df_annotation = df_annotation.drop(["proteins_id"], axis=1)

        #Convert "objects" columns to "string" (correct nan read as int)
        object_colnames = df_annotation.select_dtypes(['object']).columns
        df_annotation[object_colnames] = df_annotation[object_colnames].astype("str")

        #Convert "int64" columns to "object"
        int_colnames = df_annotation.select_dtypes(['int64']).columns
        df_annotation[int_colnames] = df_annotation[int_colnames].astype("object")
        
        #If the two dataframe have the same length, concatenate them
        if len(df_embeddings) == len(df_annotation):
            df_concat = pd.concat([df_embeddings, df_annotation], axis=1)
        else:
            raise ValueError("Number of sequence and number of annotation doesn't match.")

        df_concat.set_index(["proteins_id"], inplace=True)
        df_concat.insert(0, "proteins_id", df_concat.index)
        df_concat["proteins_id"] = df_concat["proteins_id"].astype("object")

        return df_concat
    
    else:
        df_embeddings.set_index(["proteins_id"], inplace=True)
        df_embeddings.insert(0, "proteins_id", df_embeddings.index)
        df_embeddings["proteins_id"] = df_embeddings["proteins_id"].astype("object")
        return df_embeddings


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
        labels_text=None, # list of strings: values of labels_name that we would like to mark on the plot
        labels_name=None, # string: a name of the dataframe column used for coloring and legend
        labels_idx=None,
        col_dict=None,
        title=None,
        show_lines=False,
        show_text=False,
        show_legend=True,
        axis_equal=True,
# advanced settings:
        legend_cols=5,
        circle_size=30,
        circe_transparency=1.0,
        line_transparency=0.8,
        line_width=0.8,
        fontsize=9,
        fig_width=10,
        fig_height=10,
        file_name=None,
        file_format=None,
        width_ratios=[7, 1],
        bbox=(1.3, 0.7),
        plot_legend=False,
        is_hyperbolic=True
    ):
    # TODO add readdocs
    # create data structure suitable for embedding

    if not (labels_name is None):
        labels = df[labels_name].values
    else:
        labels_name = 'labels'
        df[labels_name] = 'na'
    
    if labels_text == 'all':
        labels_text = labels


    num_labels = len(np.unique(labels))
    print("Number of unique labels: ", num_labels)

    # To avoid scale changing when changing number of labels
    # we recalculate figure size (assuming 5 legend columns)
    point_size = 0.0138889
    labelspacing = 0.5
    labelspace_height = labelspacing*point_size
    text_height = fontsize*point_size
    add_height = (int(num_labels/5)+1)*(text_height+labelspace_height) # number of lines
   

    fig = plt.figure(figsize=(fig_width, fig_height + add_height))#, constrained_layout=True)
#    gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
    ax = plt.subplot()#figsize=(fig_width, fig_height))#gs[0])
    
    sns.despine(left=False, bottom=False, right=True)
    
    # if col_dict is None:
    #     unique_labels = np.unique(labels)
    #     col_dict = dict(zip(unique_labels, ['#00000']*len(unique_labels)))
    
    if (col_dict is None) and not (labels is None):
        col_dict = get_colors(labels)

    if is_hyperbolic:
        circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
#        circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
#        circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
#        circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
#        circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
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
#        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#           ncol=2, mode="expand", borderaxespad=0.)
        plt.legend(numpoints=1, loc='upper center',
               bbox_to_anchor=(0.5, -0.01), fontsize=fontsize, ncol=5)#int(len(np.unique(labels))/20))
    
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
        labels_idx = []
        for cur_label in labels_text:
            labels_idx += list(np.where(labels == cur_label)[0])
        for i in labels_idx:
            texts.append(
                ax.text(
                    emb[i, 0], 
                    emb[i, 1], 
                    labels[i], 
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

#    plt.tight_layout()

    if file_name:
        save_to_file(fig, file_name, file_format)        

    return plt


def plot_embedding_interactive(
        df,
        labels=None,
        labels_text=None,
        labels_idx=None,
#        col_dict=None,
        title=None,
        show_lines=False,
        show_text=False,
        show_legend=True,
        circle_size=30,
        circe_transparency=1.0,
        line_transparency=0.8,
        line_width=0.8,
        fontsize=9,
#        fig_width=5,
#        fig_height=5,
        file_name=None,
        file_format=None,
        labels_name=None,
        second_labels_name=None, 
#       width_ratios=[7, 1],
#        bbox=(1.3, 0.7),
        plot_legend=False,
        color_palette=None, 
    ):
    if not (labels_name is None):
        labels = df[labels_name].values
    else:
        labels_name = 'labels'
        df[labels_name] = 'na'

    if labels_text == 'all':
        labels_text = labels

    #if (col_dict is None) and not (labels is None):
    #    col_dict = get_colors(labels)

    plt = px.scatter(df, x = 'pm1', y = 'pm2', 
                     color = labels_name, 
                     color_discrete_map=color_palette,
                     color_continuous_scale = "Turbo",
                     #labels={'color': labels}, 
                     hover_name=labels_name, 
                     hover_data=df.columns[2:], 
                     height = 800,
                     title = title)
    
    if show_text and (not (labels_text is None)): 
        if (second_labels_name is None):
            texts = df.loc[df[labels_name].isin(labels_text)]
            labels_idx = texts.index

            for i in labels_idx:
                plt.add_annotation(x=texts.loc[i,'pm1'], y=texts.loc[i,'pm2'],
                       text=texts.loc[i, labels_name],
                       showarrow=True,
                       arrowhead=1)
            plt.update_traces(textposition='top right')
        else: 
            texts = df.loc[df[second_labels_name].isin(labels_text)]
            labels_idx = texts.index
            for i in labels_idx: 
                plt.add_annotation(x=texts.loc[i,'pm1'], y=texts.loc[i,'pm2'],
                    text=texts.loc[i, second_labels_name], 
                    showarrow=True,
                    arrowhead=1)
            plt.update_traces(textposition='top right') 
  
    # calculate radii of the concentric circles with equivalent step in hyperbolic space
    
    x1 = 0.25
    hyper_rad = [x1]
    num_circ =6 
    for i in range(num_circ):
        z = 0.5*(np.cosh((i+1)*np.arccosh(1+2*x1**2/(1-x1**2)))-1)
        a = 1+z*(1-x1**2)
        b = -2*x1
        c = x1**2*(z+1)-z
#        hyper_rad.append((-b + np.sqrt(b**2 - 4*a*c))/(2*a))
        hyper_rad = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        plt.add_shape(type="circle",
                    x0=-hyper_rad, y0=-hyper_rad, x1=hyper_rad, y1=hyper_rad,
  #                  x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
                    line=dict(color="black", dash = "dot"), 
                    opacity = 0.15)
   
    plt.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-1, y0=-1, x1=1, y1=1,
                  line_color="Black")
    plt.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-0.005, y0=-0.005, x1=0.005, y1=0.005,
                  line_color="Black",
                  fillcolor="Black")
        
#    plt.add_shape(type="circle",
#                    x0=-hyper_rad[0], y0=-hyper_rad[0], x1=hyper_rad[0], y1=hyper_rad[0],
#  #                  x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
#                    line=dict(color="black", dash = "dot"), 
#                    opacity = 0.15)
    

#    plt.add_shape(type="circle",
#                    x0=-hyper_rad[1], y0=-hyper_rad[1], x1=hyper_rad[1], y1=hyper_rad[1],
##                    x0=-0.7, y0=-0.7, x1=0.7, y1=0.7,
#                    line=dict(color="black", dash = "dot"), 
#                    opacity = 0.15)
    

#    plt.add_shape(type="circle",
#                    x0=-hyper_rad[2], y0=-hyper_rad[2], x1=hyper_rad[2], y1=hyper_rad[2],
# #                   x0=-0.9, y0=-0.9, x1=0.9, y1=0.9,
#                    line=dict(color="black", dash = "dot"), 
#                    opacity = 0.15)

    plt.update_xaxes(range = [-1.1, 1.1], constrain="domain")
    plt.update_yaxes(scaleanchor = "x", scaleratio = 1)
    
    #    plt.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    plt.update_layout(yaxis2=dict(matches='y', layer="below traces", overlaying="y", ),)
    plt.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
    plt.update_layout(title_text=title, title_x=0.5, font_size=fontsize) # center the title 

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
