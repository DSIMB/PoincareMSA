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

python $prep_scripts/ali2freq-py3.py -gapaa -al $name.clean${gapth}.mfasta -m $prep_scripts/homstradfreq.txt  > $rep/root.txt
python $prep_scripts/normalize_gaps.py -i $rep/root.txt 

