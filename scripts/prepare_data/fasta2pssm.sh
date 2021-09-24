#!/bin/bash
#####################################################################################
### Calculates an amino acid profile for each N.fasta file in each protein family ###
### using homstrad substitution matrix calculated by Jean-Chrostophe              ### 
### and considering gaps as 21st amino acid.                                      ###
#####################################################################################

rep="fastas0.9"

for f in $rep/*.fasta; do
    filename=$(basename -- "$f")
    filename="${filename%.*}"
    python ../ali2freq/ali2freq-py3.py -gapaa -al $f -m ../ali2freq/homstradfreq.txt  > $rep/$filename.txt
done
