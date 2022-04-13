# from: https://biopython.org/wiki/Phylo_cookbook
# https://bioinfo-fr.net/automatiser-le-parcours-et-la-manipulation-darbres-phylogenetiques-avec-le-module-bio-phylo-de-biopython

from Bio import Phylo
import numpy

def to_adjacency_matrix(tree):
    """Create an adjacency matrix (NumPy array) from clades/branches in tree.

    Also returns a list of all clades in tree ("allclades"), where the position
    of each clade in the list corresponds to a row and column of the numpy
    array: a cell (i,j) in the array is 1 if there is a branch from allclades[i]
    to allclades[j], otherwise 0.

    Returns a tuple of (allclades, adjacency_matrix) where allclades is a list
    of clades and adjacency_matrix is a NumPy 2D array.
    """
    allclades = list(tree.find_clades(order='level'))
    lookup = {}
    for i, elem in enumerate(allclades):
        lookup[elem] = i
    adjmat = numpy.zeros((len(allclades), len(allclades)))
    for parent in tree.find_clades(terminal=False, order='level'):
        for child in parent.clades:
            adjmat[lookup[parent], lookup[child]] = 1
    if not tree.rooted:
        # Branches can go from "child" to "parent" in unrooted trees
        adjmat = adjmat + adjmat.transpose()
    return (allclades, numpy.matrix(adjmat))



def to_distance_matrix(tree):
    """Create a distance matrix (NumPy array) from clades/branches in tree.

    A cell (i,j) in the array is the length of the branch between allclades[i]
    and allclades[j], if a branch exists, otherwise infinity.

    Returns a tuple of (allclades, distance_matrix) where allclades is a list of
    clades and distance_matrix is a NumPy 2D array.
    """
    allclades = list(tree.find_clades(order='level'))
    lookup = {}
    for i, elem in enumerate(allclades):
        lookup[elem] = i
    distmat = numpy.repeat(numpy.inf, len(allclades)**2)
    distmat.shape = (len(allclades), len(allclades))
    for parent in tree.find_clades(terminal=False, order='level'):
        for child in parent.clades:
            if child.branch_length:
                distmat[lookup[parent], lookup[child]] = child.branch_length
    if not tree.rooted:
        distmat += distmat.transpose()
    return (allclades, numpy.matrix(distmat))




tree = Phylo.read("../DATA/mmo/mmo.phylip_phyml_tree.txt", "newick")
#allclades, adjmat = to_adjacency_matrix(tree)
#print adjmat 
#print allclades

#print(tree.depths())
resultat = tree.find_clades(lambda c : c.branch_length < 1)
print(resultat.next())
#print(tree.find_any({'name' : '1'}))
