import glob
import sys

def get_num_lines(fname):
    count = 0
    with open(fname, 'r') as f:
        for line in f:
            count += 1
        return count


flist = glob.glob(sys.argv[1])
maxval = 0
for fname in flist:
    maxval = max(maxval, get_num_lines(fname))
print(maxval)
