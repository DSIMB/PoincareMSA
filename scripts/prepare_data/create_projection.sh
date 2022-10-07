#!/usr/bin/bash

#################################################
### Step 1: Prepare files starting from fasta ###
#################################################

export MAX_CPUS=$(getconf _NPROCESSORS_ONLN) # for parallelization
export prep_scripts="$1" # directory containing scripts
export mfasta="$2" # pathway to .mfasta file
export path_out="$3"
export out_name="$4"
export gapth="$5"

echo "Input file: ${mfasta}"
name=`basename $mfasta .mfasta`
echo "Name of the protein family: ${name}"


#prep_scripts="../../scripts/prepare_data" # directory containing scripts

#gapth=0.9 # threshold for gaps filtering
mkdir -p $path_out
export rep=$path_out/"fasta${gapth}" # output directory to contain encodings per sequence
mkdir -p $rep

# calculate weights of every sequence in the alignment
python $prep_scripts/compute_weight_sequence-py3.py -al $mfasta > $path_out/$out_name.seq_weight
# filter sequence positions according to a given threshold on weighted gaps frequency  
python $prep_scripts/filter_gaps.py -i $mfasta -w $path_out/$out_name.seq_weight -o $path_out/$out_name.clean${gapth}.mfasta -g $gapth
# split clean mfasta file to indifidual fasta files per sequence
python $prep_scripts/mfasta2fasta.py -i $path_out/$out_name.clean${gapth}.mfasta -o $rep

# Calculates an amino acid profile for each equence (fasta file) using homstrad substitution matrix calculated by Jean-Chrostophe Gelly. 
# Gaps are considered as 21st amino acid. 20 columns are then divided by the 21st (gap) column for normalisation.

translate(){
    filename=$(basename -- "$1")
    filename="${filename%.*}"
#    echo $filename
#    prep_scripts="$2"
#    rep="$3"
    #echo $1
    #echo $prep_scripts
    #echo $rep
    python $prep_scripts/ali2freq-py3.py -gapaa -al $1 -m $prep_scripts/homstradfreq.txt  > $rep/$filename.txt
    python $prep_scripts/normalize_gaps.py -i $rep/$filename.txt -o $rep/$filename.aamtx 
}
export -f translate

#ls $rep/*.fasta > $path_out/list_fasta
#echo $MAX_CPUS
#cat list_fasta | xargs -P $MAX_CPUS -l bash -c 'translate $1'
#echo translate "$0" "$prep_scripts" "$rep"
#find $rep -type f -name '*.fasta' | xargs -P 1 -l bash -c 'translate "$0" "$prep_scripts" "$rep"'
find $rep -type f -name '*.fasta' | xargs -P $MAX_CPUS -l bash -c 'translate "$0"'
#"$prep_scripts" "$rep"'

#for f in $rep/*.fasta; do
#    filename=$(basename -- "$f")
#    filename="${filename%.*}"
#    python $prep_scripts/ali2freq-py3.py -gapaa -al $f -m $prep_scripts/homstradfreq.txt  > $rep/$filename.txt
#    python $prep_scripts/normalize_gaps.py -i $rep/$filename.txt -o $rep/$filename.aamtx 
#done

# Commands to create a root point for eventual rotation of the projection
python $prep_scripts/ali2freq-py3.py -gapaa -al $mfasta -m $prep_scripts/homstradfreq.txt  > $rep/0.txt
python $prep_scripts/normalize_gaps.py -i $rep/0.txt -o $rep/0.aamtx

