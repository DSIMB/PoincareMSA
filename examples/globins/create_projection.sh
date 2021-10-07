#!/usr/bin/bash


#################################################
### Step 1: Prepare files starting from fasta ###
#################################################
mfasta="$1"
echo "input file"
echo $mfasta
name=`basename $mfasta .mfasta`
echo "name"
echo $name 

prep_scripts="../../scripts/prepare_data" # directory containing scripts
data="../../data/knottins" # directory containig .mfasta file and eventual color schemes

gapth=0.9 # threshold for gaps filtering
rep="fasta${gapth}" # output directory to contain encodings per sequence
echo $rep
echo $gapth

# calculate weights of every sequence in the alignment
python $prep_scripts/compute_weight_sequence-py3.py -al $name.mfasta > $name.seq_weight
# filter sequence positions according to a given threshold on weighted gaps frequency  
python $prep_scripts/filter_gaps.py -i $name.mfasta -w $name.seq_weight -o $name.clean${gapth}.mfasta -g $gapth
# split clean mfasta file to indifidual fasta files per sequence
mkdir $rep
python $prep_scripts/mfasta2fasta.py -i $name.clean${gapth}.mfasta -o $rep

# Calculates an amino acid profile for each equence (fasta file) using homstrad substitution matrix calculated by Jean-Chrostophe Gelly. 
# Gaps are considered as 21st amino acid. 20 columns are then divided by the 21st (gap) column for normalisation.
for f in $rep/*.fasta; do
    filename=$(basename -- "$f")
    filename="${filename%.*}"
    python $prep_scripts/ali2freq-py3.py -gapaa -al $f -m $prep_scripts/homstradfreq.txt  > $rep/$filename.txt
    python $prep_scripts/normalize_gaps.py -i $rep/$filename.txt 
done


