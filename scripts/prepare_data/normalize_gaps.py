import argparse
from numpy import *

# f - complete pathway to the file
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', dest= "in_file", type = str)
	parser.add_argument('-o', dest= "out_file", type = str)
	args = parser.parse_args()
	return (args.in_file, args.out_file)

args=get_args()
x = loadtxt(args[0])
#d = "".join(f.split("/")[:-1])
#f = f.split("/")[-1].split(".")[0]
#F = open(d+"/"+f+".aamtx", "w")

F = open(args[1], "w")
for i in range(len(x)):
  for j in range(20):
    F.write("%.4f "%(x[i,j]/sqrt(x[i,20])))
  F.write('\n')
F.close()

