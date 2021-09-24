#! /usr/bin/env python
# -*- coding: ISO-8859-1 -*-

# AUTHOR GHOUZAM YASSINE
# MASTER 2 in COMPUTATIONAL BIOLOGY AND BIOINFORMATICS
# UNIVERSITE PARIS DIDEROT

# THIS PROGRAMM GENERATE WEIGHTING SCHEMES
# FROM A MULTIPLE ALIGNMENT
# PIR FORMAT OR FASTA FORMAT


import sys
try :
    import numpypy as np
except :
    import numpy as np

import argparse

AA = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S',
'R', 'T', 'W', 'V', 'Y', '-']

def ali_long_pir(filename):
    '''Alignment length in line number'''
    f = open(filename)
    l = f.readline()
    
    flag=False
    cpt=0
    while l!="":
        if flag and l[0]=='>':
            f.close()
            break
        if l[0]=='>':
            flag=True
        cpt+=1
        l = f.readline()
    return cpt-2


def pir2list(filename):
    '''Parsing of pir alignment in list of list'''
    
    f = open(filename)
    l=f.readline()
    ali_longeur=ali_long_pir(filename)
    total=[]
    while l!="" :
        if l[0]=='>':
            l=f.readline()
            l=f.readline()
            un_ali = ""
            for i in range(ali_longeur):
                un_ali += l.replace("\n","")
                l=f.readline()
            total.append(un_ali[:-1])
    f.close()
    return total


def ali_long_fasta(filename):
    '''Alignment length in line number'''
    f = open(filename)
    l = f.readline()
    
    flag=False
    cpt=0
    while l!="":
        if flag and l[0]=='>':
            f.close()
            break
        if l[0]=='>':
            flag=True
        cpt+=1
        l = f.readline()
    return cpt-1


def fasta2list(filename):
    '''Parsing of fasta alignment in list of list'''

    f = open(filename)
    l = f.readline()
    ali_longeur = ali_long_fasta(filename)
    total = []
    while l != "" :
        if l[0] == '>':
            l = f.readline()
            un_ali = ""
            for i in range(ali_longeur):
                un_ali += l.replace("\n","")
                l = f.readline()
            total.append(un_ali)
    f.close()
    return total


def get_ali_format(filename):
    """Guess file format of the multiple alignment
    (Two formats supported)"""
    msg = "ERROR : %s UNKNOWN OR UNSUPPORTED FORMAT FILE\
    \nSupported format : PIR and FASTA"%filename
    f = open(filename,'r')
    ligne1 = f.readline()
    ligne2 = f.readline()
    f.close()
    condition1 = ligne1[0] == '>' # and ligne1[3] == ';' # commented by Tatiana: not true, if the name is very short
    condition2 = ':' in ligne2
    if condition1 and condition2 :
        # file PIR
        return "PIR-FORMAT"
    elif (((not condition1) and (not condition2)) 
        or (condition1 and (not condition2))):
        if '>' in ligne1:
            # file fasta
            return "FASTA-FORMAT"
        else :
            sys.stderr.write(msg+'\n')
            sys.exit(1) 
    else :
        sys.stderr.write(msg+'\n')
        sys.exit(1)


def ali2list(filename):
    """ Record alignment as list """
    
    format_ali = get_ali_format(filename)
    
    if format_ali == "PIR-FORMAT":
        return pir2list(filename)
    return fasta2list(filename)


def occ_pos(ali,pos):
    """aa occurence in the line"""
    occ_aa = [0]*21 # list of occ aa in order
    for seq in ali:
        occ_aa[AA.index(seq[pos])] += 1
    return occ_aa


def freq_pos(ali,pos):
    """aa frequency at a given position"""
    occ = occ_pos(ali, pos)
    freq_aa = [0]*21
    for i in range(len(occ)):
        freq_aa[i] = occ[i]/float(len(ali))
    return freq_aa


def freq_prof(ali):
    nb_pos=len(ali[0])
    return np.array([freq_pos(ali,i) for i in range(nb_pos)])


def occ_prof(ali): # a table Nx21 with number of each aa appeared (including gaps)
    nb_pos=len(ali[0])
    return np.array([occ_pos(ali,i) for i in range(nb_pos)])



def Rc2(mat_counts):
    """ Number of different aa at position (including gaps) """ 
    R = []
    for i in range(mat_counts.shape[0]):
        Rc = 0
        for j in range(mat_counts.shape[1]):
            if mat_counts[i,j] != 0:
                Rc += 1
        R.append(Rc)
    return np.array(R)


def car_in_alipos(ali, pos, car):
    """ Returns True if car is in alignment ali in position pos """
    for i in ali:
        if i[pos] == car:
            return True
    return False


def only_car_in_alipos(ali, pos, car):
    """ Returns trus if ali contains only car in position pos """
    for i in ali:
        if i[pos] != car:
            return False
    return True


def filter_ali_car(ali, car):
    """ Returns positions of the alignment not containing car """
    handled_pos = [i for i in range(len(ali[0])) 
    if not car_in_alipos(ali, i, car)]
    
    filtered_ali = []
    for i in range(len(ali)):
        filtered_seq = ''
        for j in handled_pos:
            filtered_seq += ali[i][j]
        filtered_ali.append(filtered_seq)
    return filtered_ali


def filter_ali_car_threshold(ali, car, threshold):
    """ Removes the positions of alignment with car 
    present in amount above the threshold %"""
    handled_pos = [i for i in range(len(ali[0])) if freq_pos(ali, i)[AA.index(car)] <= threshold ]
#    print handled_pos
    filtered_ali = []
    for i in range(len(ali)):
        filtered_seq = ''
        for j in handled_pos:
            filtered_seq += ali[i][j]
        filtered_ali.append(filtered_seq)
    return filtered_ali


def filter_ali_onlycar(ali, car):
    """ Remove positions of alignment containing only car """
    handled_pos = [i for i in range(len(ali[0])) if not only_car_in_alipos(ali, i, car)]
    filtered_ali = []
    for i in range(len(ali)):
        filtered_seq = ''
        for j in handled_pos:
            filtered_seq += ali[i][j]
        filtered_ali.append(filtered_seq)
    return filtered_ali


def contains_non_AA(seq, AA):
    """ Returns True if the sequence contains elements other than AA """
    for i in seq :
        if i not in AA:
            return True
    return False


def filter_ali_seq_gap(ali, seuil, AA):
    """ Removes sequences with amount of gaps (of elements not AA) above the threshold % """
    queryseq = ali[0]
    n = len(queryseq)
    for i in range(n):
        if queryseq[i] not in AA:
            ali[0][i] = '-'
    
    res = [i for i in ali[1:] if ( (i.count("-")/(n*1.0)) < seuil and not contains_non_AA(i,AA) ) ]
    res.insert(0,ali[0])
    return res


def read_profile(filename):
    """ Reads a profile from file """
    fichier = open(filename,"r")    # pointeur sur 1er ligne fichier
    return np.array([line[:-1].split() for line in fichier], dtype=float)



def get_args():
    
    usage = "\nali2freq.py -al alignment_file -m matrix -gts gap_threshold_seq\
    (default 100 %%) -gtc gap_threshold_column (default 100 %%)"
    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument('-al', dest = "ali_file", type = str, help = "File\
    of multiple alignment (Pir of fasta format)")
    parser.add_argument('-gts', dest = "gap_threshold_seq", default = 100,
    type = float, help = "Sequences with a percentage of gaps superior than the\
    gap_threshold were not considered" )
    parser.add_argument('-gtc', dest = "gap_threshold_col", default = 100, 
    type = float, help = "Columns with a percentage of gaps superior than the\
        gap_threshold_col were not considered for the weighting scheme" )
    parser.add_argument('-v', dest = "verbose", action = "store_true",
    help = "Verbose mode")
    args = parser.parse_args()
    
    return (args.ali_file, args.gap_threshold_seq/100.0
    , args.gap_threshold_col/100.0, args.verbose)

def X_to_gap(ali, AA):
    """ Replaces all unknown characters by gap """
    queryseq = ali[0]
    n = len(queryseq)
    ali_out = []
    nx = 0
    for j in range(len(ali)):
        seq = ""
        for i in range(n):
            if ali[j][i] not in AA:
                seq = seq+'-'
                nx += 1
            else:
                seq=seq+ali[j][i]
        ali_out.append(seq)
#	res = [i for i in ali[1:] if ( (i.count("-")/(n*1.0)) < seuil
#	and not contains_non_AA(i,AA) ) ]
#	res.insert(0,ali[0])
#	print >> sys.stderr, "%d X aa replaced by gaps in %d sequences"%(nx, len(ali))
    print("%d X aa replaced by gaps in %d sequences"%(nx, len(ali)), file=sys.stderr)
    return ali_out

########################################
########################################
##-------------- MAIN-----------------##
########################################
########################################

## Reading input ##
ali_file, gap_threshold_seq, gap_threshold_col, verbose = get_args()
alignts = ali2list(ali_file) # input multiple alignment

num_init_alignts=len(alignts)
num_pos_init_alignts=len(alignts[0])
alignts = X_to_gap(alignts, AA)
## Preprocessing (if no option -gaps is on) ##

# We remove sequences containing more than 70% of gaps or unknown characters
alignts = filter_ali_seq_gap(alignts, gap_threshold_seq, AA)
# We remove positions containing only gaps 
alignts = filter_ali_onlycar(alignts, '-')
# And finally remove positions with number of gaps above the threshold
alignts_g_f = filter_ali_car_threshold(alignts, '-', gap_threshold_col) 

if verbose:
#    print 'alignments after removing gapped sequences and all-gap positions: ', alignts
#    print 'alignments after filtering highly gapped positions (for weighting): ', alignts_g_f
    print >> sys.stderr, "Number of sequence after fitlering:  %d/%d" %( len(alignts_g_f),num_init_alignts ) 
    print >> sys.stderr, "Number of position in sequence after fitlering:  %d/%d" %( len(alignts_g_f[0]),num_pos_init_alignts ) 

########################################################
# Weighting scheme de J.G Henikoff et S. Henikoff 1994 #
########################################################

filtered_occ = occ_prof(alignts_g_f) # occurences for sequence weighting

Rc_without_g = Rc2(filtered_occ) # number of different aa in position c (remaining after filtering gaps are taken into account!)
# print "Rc_without_g: ", Rc_without_g

Norm_weights = []
nb_pos = len(alignts_g_f[0])

for i in alignts_g_f:
    Total = 0
    for j in range(len(i)):
        Total += 1.0/( Rc_without_g[j] * filtered_occ[j, AA.index(i[j])])
    print('%4.8f'%(Total/nb_pos))
#        Norm_weights.append(Total/nb_pos)
