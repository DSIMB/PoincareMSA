#!/bin/bash
rep="fastas0.9"
rm $rep/*.aamtx
for f in $rep/*.fasta; do
	filename=$(basename -- "$f")
	filename="${filename%.*}"
	echo "$rep", "$filename", $f
	python ../PoincareMaps-DataPreparation/from-mfasta/normalize_gaps.py -i $rep/$filename.txt 
	#./../ali2freq_query_tatiana.py -al $f -m ../../homstradfreq.txt -gts 100 -gtc 100  > $rep/$filename.txt
done

