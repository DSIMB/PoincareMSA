#!/bin/bash
path=../DATA
#export path=../DATA
#for rep in ../DATA/AAA ; do
while IFS= read -r rep; do
	s1=$(basename $rep)
	for j in 1 2 3 4 5
	do
		python ./TreeCluster-master/TreeCluster.py -i $path/$rep/$s1.phylip_phyml_tree.txt -o $path/$rep/"$s1"_tree_cluster_"$j".out -t "$j"	
		python create_dict.py -i $path/$rep/"$s1"_tree_cluster_"$j".out -o $path/$rep/"$s1"_tree_cluster_"$j".pkl
	done

done < "/scratch/galochkina/HOMSTRAD_DATA/pdb_mfasta_for_projection/my_scripts/list_selected.new"

