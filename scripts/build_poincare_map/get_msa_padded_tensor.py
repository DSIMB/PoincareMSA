import numpy as np
from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--path", type=str,
        required=True,
        help="Path of aatmx file"
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=872,
        help="Max length",
    )
    return parser

# Construct a padded numpy matrices for a given PSSM matrix
def construct_tensor(fpath, maxlen):
    arr = np.loadtxt(fpath)
    ansarr = np.zeros((maxlen, 20))
    ansarr[:arr.shape[0], :] = arr
    return ansarr

# L2 norm between 2 matrices
def l2_distance(a, b):
    return np.linalg.norm(a - b)

def main():
    parser = create_parser()
    args = parser.parse_args()
    a = construct_tensor(args.path, args.maxlen)
    b = construct_tensor(args.path, args.maxlen)
    dist = l2_distance(a, b)
    assert dist == 0.0

if __name__ == "__main__":
    main()
