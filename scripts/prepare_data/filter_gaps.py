#######################################################################
### Takes as input .mfasta file and removes sequences containing X. ###
#######################################################################
from numpy import *
import argparse

AA = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S',
'R', 'T', 'W', 'V', 'Y', '-']

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', dest= "f", type = str, help = "input alignment to filter")
	parser.add_argument('-w', dest= "w", type = str, help = "weights calculated from the alignment")
	parser.add_argument('-o', dest= "o", type = str, help = "output file")
	parser.add_argument('-g', dest= "g", type = float, default=0.7, help = "threshold for the weighted gap frequency")
	args = parser.parse_args()
	return (args.f, args.g, args.w, args.o)

def seq2vec(seq): # translate sequence to 0 for a.a. and 1 for gap
	vec = zeros(len(seq))
	for i in range(len(seq)):
		if seq[i]=="-":
			vec[i]=1
	return vec

def X_to_gap_seq(seq, AA):
	""" Replaces all unknown characters by gap """
	n = len(seq)
	nx = 0
	seq_new = ""
	for i in range(n):
		if seq[i] not in AA:
			seq_new = seq_new + '-'
			nx += 1
		else:
			seq_new = seq_new + seq[i]
#	res = [i for i in ali[1:] if ( (i.count("-")/(n*1.0)) < seuil
#	and not contains_non_AA(i,AA) ) ]
#	res.insert(0,ali[0])
#	print >> sys.stderr, "%d X aa replaced by gaps in %d sequences"%(nx, len(ali))
	return seq_new

############
### MAIN ###
############

# if the score of gaps in a position is > 0.7, we eliminate the corresponding position
inp, g, weights, out = get_args() # f - file name, g - gaps threshold for filter

weight = loadtxt(weights) # calculated with compute_weight_sequence_position.pl 
F = open(inp, "r") # new .mfasta file with all sequences containing X eliminated
M = [] 
MS = []
names = []

line = F.readline()

while line!="":
  lines=""
  names.append(line) # >name of the sequence
  line = F.readline()
  while (line != "") and (">" not in line):
    lines=lines+line
    line = F.readline()
  lines = lines.replace("\n","")
  lines = X_to_gap_seq(lines, AA)
  MS.append(lines)
  M.append(seq2vec(lines))
F.close()

M = array(M) # M contains list of sequences translated in 0 and 1 profiles with 1 for gaps
Msum = dot(M.T,weight) # weighted sum of gaps

FN = open(out, "w") # new .mfasta file with all sequences containing X eliminated

# write down only positions with gap score below threshold
for i in range(len(MS)):
  FN.write(names[i])
  pos = 0
  for j in range(len(Msum)):
    if pos==60:
      FN.write("\n")
      pos=0
    if Msum[j] < g:
      FN.write(MS[i][j])
      pos+=1
  FN.write("\n")

FN.close()

print("filter_gaps finished for", inp)

