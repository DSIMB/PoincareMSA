PoincaréMSA is a tool for protein family vizualisation. It takes as input a multiple sequence alignemnt (MSA) and builds its projection on a Poincaré disk using the method developed by Klimovskaia et al. in [1].

## General description
PoincareMSA relies on hyperbolic projections build as described in the original paper. The proximity of the points to the disk center corresponds to their hierarchy and correlates with the proximity of the proteins to the root of the phylogenetic tree. Thus, must central point often correspond to the ancestor proteins and protein located close to the border -- to the leaves of phylogenetic tree. The projection also preserves local proximity of sequences and the observed groups correspond to proteins sharing similar properties.

## Colab version
We provide an interactive Google Colab notebook for projection of protein MSA. 

## Version for local installation
The program is implemented in `python` using `pytorch` librarie for Poincaré disk construction and `plotly` for interactive visualisation of the resulting projections.

One can use a conda environment to access all the cenessary libraries (UPD: yml file seems to work only on ubuntu):
```
conda env create -f env_poincare.yml
conda activate env_poincare
```
### Python notebooks

The best way to try PoincaréMSA is by launching python notebooks with provided examples. Here goes the protocol how to launch them ...

Otherwise, the used can also launche the projection generation step by step as described below.

### Command line step-by-step bersion

Data preparation

The user is invited to provide its MSA in the classical `.mfasta` format. Each sequence of the alignment is translated to a profile using position-specific scoring matrix (PSSM) according to the pseudo-count algortihm of Henikoff & Henikoff. The related scripts are located in `scripts/prepare_data/` directory and driver scripts are provided for every example as `create_projection.sh`.

### Poincaré projection implementation

The resulting PSSM profiles representing each protein of MSA are stored in the directory `fasta0.9`, where 0.9 indicate the threshold percentage of gaps per position used to filter initial alignment. To build a Poincaré disk from this data, one needs to run a command from `scripts/build_poincare_map/` directory:

```
python main.py --input_path /home/lavande/galochkina/SCIENCE/POINCARE/PoincareMSA/examples/globins/fasta0.7/ --output_path /home/lavande/galochkina/SCIENCE/POINCARE/Globins/TEST/  --knn 5 --gamma 4 --batchsize 4 --rotate --epochs 1000
python main.py --input_path path_to_input_dir/PoincareMSA/examples/thioredoxins/fasta0.9/ --output_path path_to_out_dir 
```
which will create an output `.csv` file with protein coordinates in the final projection and `.png` images reflecting the learning process. The `.csv` file can be further used to build interactive visualisation.

### Visulaisation
The easiest way to visualise your results is by using interactive representation implemented in python notebook. Examples of such notebooks are provided for every protein family considered here. The user can annotate and color the projected points according to any annotaion provided as a `.csv` file and having number of rows and order of rows corresponding to the number and order of sequences in the initial MSA.

### Examples of use
We provide several examples of PoincareMSA usage for different protien families in the `examples` folder. Each example comes with a bash script alloqing to reproduce the results starting from MSA and labels contained in `data`.

### References
The projection construction is adapted from the original code: https://github.com/facebookresearch/PoincareMaps

[1] Klimovskaia, A., Lopez-Paz, D., Bottou, L. et al. Poincaré maps for analyzing complex hierarchies in single-cell data. Nat Commun 11, 2966 (2020). https://doi.org/10.1038/s41467-020-16822-4

