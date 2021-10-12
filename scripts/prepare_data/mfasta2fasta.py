##################################################################################
### Takes as input .mfasta file and splits it in N.fasta files for each entry. ### 
### Also writes a summary file with N and protein Id.                          ###
##################################################################################

import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', dest= "f", type = str, help = "alignment file to split")
	parser.add_argument('-o', dest= "d", type = str, help = "output directory")
	args = parser.parse_args()
	return (args.f, args.d)

f, path = get_args()

F = open(f, "r") # no X and gaps filtered
S  = open(f.split(".mfasta")[0] + ".sum", "w") # file with number-protein id correspondence
E = open(f.split(".mfasta")[0] + ".NUM.mfasta", "w")

i = 1 # iteration over sequences
line = F.readline()

while line!="":
  G = open(path + "/" + str(i) + ".fasta", "w")
  E.write('>'+str(i)+'\n')
  G.write(line)
  S.write(str(i)+"\t"+line)
  line = F.readline()
  while (line != "") and (">" not in line):
    G.write(line)
    E.write(line)
    line = F.readline()
  G.close()
  i+=1

  
print("mfasta2fasta finished for ", f)
F.close()
S.close()
E.close()
