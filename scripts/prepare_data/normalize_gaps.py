import argparse
from numpy import *

# f - complete pathway to the file
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', dest= "f", type = str)
	args = parser.parse_args()
	return (args.f)

f=get_args()
x = loadtxt(f)
d = "".join(f.split("/")[:-1])
f = f.split("/")[-1].split(".")[0]
F = open(d+"/"+f+".aamtx", "w")

for i in range(len(x)):
  for j in range(20):
    F.write("%.4f "%(x[i,j]/sqrt(x[i,20])))
  F.write('\n')

F.close()

