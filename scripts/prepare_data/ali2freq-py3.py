#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

# AUTHOR GHOUZAM YASSINE
# MASTER 2 in COMPUTATIONAL BIOLOGY AND BIOINFORMATICS
# UNIVERSITE PARIS DIDEROT

# THIS PROGRAMM GENERATE A PSSM 
# (+ PSEUDO-COUNTS, WEIGHTING SCHEMES) FROM A MULTIPLE ALIGNMENT
# PIR FORMAT OR FASTA FORMAT


import sys
try :
    import numpypy as np
except :
    import numpy as np

import argparse

AA = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S',
'R', 'T', 'W', 'V', 'Y', '-']

BG_FREQ_BLOSUM62 =   {'A':0.0812,'C':0.0130,'E':0.0668,'D':0.0591,'G':0.0741,'F':0.0395,
'I':0.0552,'H':0.0269,'K':0.0577,'M':0.0229,'L':0.0893,'N':0.0447,'Q':0.0383,
'P':0.0466,'S':0.0624,'R':0.0499,'T':0.0522,'W':0.0142,'V':0.0680,'Y':0.0350}

BG_FREQ_HOMSTRAD62 = {'A':0.0751,'R':0.0451,'N':0.0395,'D':0.0527,'C':0.0129,'Q':0.0337,
'E':0.0590,'G':0.0683,'H':0.0206,'I':0.0516,'L':0.0802,'K':0.0533,'M':0.0191,
'F':0.0364,'P':0.0422,'S':0.0518,'T':0.0502,'W':0.0123,'Y':0.0324,'V':0.0637,'-':0.1000}


MTXCAR_NOGAPS = "ARNDCQEGHILKMFPSTWYV"
MTXCAR_GAPS   = "ARNDCQEGHILKMFPSTWYV-"

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
    condition1 = ligne1[0] == '>'# and ligne1[3] == ';'
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


def qia(a, i, mtx) :
    '''Score beetween a and i in the mtx '''
    return mtx[MTXIND[a],MTXIND[i]]


def Rc1(mat_counts):
    """  Number of different aa at position (excluding gaps) """ 
    R = []
    for i in range(mat_counts.shape[0]):
        Rc = 0        
        for j in range(mat_counts.shape[1]-1):
            if mat_counts[i,j] != 0:
                Rc += 1
        R.append(Rc)
    return np.array(R)


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
    handled_pos = [i for i in range(len(ali[0])) if freq_pos(ali, i)[AA.index(car)] <= threshold]
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
#    res = [i for i in ali[1:] if ( (i.count("-")/(n*1.0)) < seuil
#    and not contains_non_AA(i,AA) ) ]
#    res.insert(0,ali[0])
#    print >> sys.stderr, "%d X aa replaced by gaps in %d sequences"%(nx, len(ali))
    print("%d X aa replaced by gaps in %d sequences"%(nx, len(ali)), file=sys.stderr)
    return ali_out

def bca(Bc, a, Nc, raw_occ_c, mtx_freq):
    A = 0    
    for i in MTXCAR:
        nci = raw_occ_c[AA.index(i)]
        qia_value = qia(a, i, mtx_freq)
        Qi = BG_FREQ[i]
        A += (nci/Nc) * (qia_value/Qi)
    return Bc * A


def occ_weighted_pos(ali, pos, Norm_weights):
    """ Returns occurence of aa in position pos weighted by w """
    occ_aa = [0]*21
    for i in range(len(ali)):
        occ_aa[AA.index(ali[i][pos])] += Norm_weights[i]
    return occ_aa


def occ_prof_weighted(ali, Norm_weights):
    nb_pos=len(ali[0])
    return (np.array([occ_weighted_pos(ali,i, Norm_weights)
    for i in range(nb_pos)]))


def read_profile(filename):
    """ Reads a profile from file """
    fichier = open(filename,"r")    # pointeur sur 1er ligne fichier
    return np.array([line[:-1].split() for line in fichier], dtype=float)


def conserve_col_of_queryseq(ali):
        queryseq = ali[0]
        pos_of_query = [i for i in range(len(queryseq)) if queryseq[i] != "-"]
        res = []
        for i in ali:
                unali = ""
                for j in pos_of_query : 
                         unali += i[j]
                res.append(unali)
        return res



def get_args():
    
    usage = "\nali2freq.py -al alignment_file -m matrix -gts gap_threshold_seq\
    (default 100 %%) -gtc gap_threshold_column (default 100 %%)"
    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument('-al', dest = "ali_file", type = str, help = "File\
    of multiple alignment (Pir of fasta format)")
    parser.add_argument('-m', dest = "matrix_file", type = str, help = 
    "Matrix of substitution for the pseudo-counts" )
    parser.add_argument('-gts', dest = "gap_threshold_seq", default = 100,
    type = float, help = "Sequences with a percentage of gaps superior than the\
    gap_threshold were not considered" )
    parser.add_argument('-gtc', dest = "gap_threshold_col", default = 100, 
    type = float, help = "Columns with a percentage of gaps superior than the\
        gap_threshold_col were not considered for the weighting scheme" )
    parser.add_argument('-first', dest = "first", action = "store_true",
    help = "Remove all positions corresponding to the gaps\
    in the first (query) sequence" )
    parser.add_argument('-gapaa', dest = "gaps", action = "store_true",
    help = "Consider gap as 21st aa for profile construction\
    using 21 background frequencies (HOMSTRAD)" )
    parser.add_argument('-gapc', dest = "gapc", action = "store_true",
    help = "Add 21th column with gaps frequency profile" )
    parser.add_argument('-v', dest = "verbose", action = "store_true",
    help = "Verbose mode")
    args = parser.parse_args()
    
    return (args.ali_file, args.matrix_file, args.gap_threshold_seq/100.0
    , args.gap_threshold_col/100.0, args.first, args.gaps, args.gapc, args.verbose)


########################################
########################################
##-------------- MAIN-----------------##
########################################
########################################

## Reading input ##
ali_file, matrix_file, gap_threshold_seq, gap_threshold_col, first, gaps, gapc, verbose = get_args()

if (gapc & gaps):
    msg = 'ERROR: You cannot call both -gapc and -gapaa.\
        \nIf wont to calculate gaps frequency from pseudocounts, use -gapc.\
        \nIf want to consider gaps as aa and to use substitution matrix 21x21, use -gapaa.'
    sys.stderr.write(msg+'\n')
    sys.exit(1)

alignts = ali2list(ali_file) # input multiple alignment
mtxfreq = read_profile(matrix_file) # substitution matrix

## Choose the regime ## 
if gaps: # gap as aa, substiution matrix 21x21
    BG_FREQ = BG_FREQ_HOMSTRAD62
    MTXCAR = MTXCAR_GAPS
else:
    BG_FREQ = BG_FREQ_BLOSUM62
    MTXCAR = MTXCAR_NOGAPS

MTXIND = {}
for i in MTXCAR :
    MTXIND[i] = MTXCAR.index(i)


num_init_alignts=len(alignts)
num_pos_init_alignts=len(alignts[0])

## Replace all unknown characters by '-'
alignts = X_to_gap(alignts, AA)
## Preprocessing (if no option -gaps is on) ##
#if gaps:
#    alignts_g_f = alignts
#else:

# We remove sequences containing more than 70% of gaps or unknown characters
alignts = filter_ali_seq_gap(alignts, gap_threshold_seq, AA)
# We remove positions containing only gaps 
if not gaps:
    alignts = filter_ali_onlycar(alignts, '-')
# And finally remove positions with number of gaps above the threshold
alignts_g_f = filter_ali_car_threshold(alignts, '-', gap_threshold_col) 

if verbose:
#    print 'alignments after removing gapped sequences and all-gap positions: ', alignts
#    print 'alignments after filtering highly gapped positions (for weighting): ', alignts_g_f
    print >> sys.stderr, "Number of sequence after fitlering:  %d/%d"%(len(alignts_g_f), num_init_alignts) 
    print >> sys.stderr, "Number of position in sequence after fitlering:  %d/%d"%(len(alignts_g_f[0]), num_pos_init_alignts) 

## Remove all columns corresponding to gaps in the first sequence if required ##
if first:
    alignts_query_length = conserve_col_of_queryseq(alignts) # even if we have filtered some position on car_threshold, we want to keep alignment length equal to that of the sequence 
else:
    alignts_query_length = alignts_g_f 

########################################################
# Weighting scheme de J.G Henikoff et S. Henikoff 1994 #
########################################################

filtered_occ = occ_prof(alignts_g_f) # occurences for sequence weighting
#filtered_occ = occ_prof(alignts_query_length) # occurences for sequence weighting

if gaps:
    Rc_without_g = Rc2(filtered_occ) # number of different aa in position c (remaining after filtering gaps are taken into account!)
else:
    Rc_without_g = Rc1(filtered_occ) # number of different aa in position c (remaining after filtering gaps are taken into account!)
# print "Rc_without_g: ", Rc_without_g

Norm_weights = []
nb_pos = len(alignts_g_f[0])
#nb_pos = len(alignts_query_length[0])

for i in alignts_g_f: # alignts_query_length
    Total = 0
    for j in range(len(i)):
        Total += 1.0/( Rc_without_g[j] * filtered_occ[j, AA.index(i[j])])
    Norm_weights.append(Total/nb_pos)

# print 'Norm_weights: ', Norm_weights



#####################################################
# Pseudo-counts de J.G Henikoff et S. Henikoff 1996 #
#####################################################

occ_wght = occ_prof_weighted(alignts_query_length, Norm_weights) # calculate occurency using weights
raw_occ = occ_prof(alignts_query_length)  # occurencies in the initial profile (columns gapped in the first sequence removed if option -first)

if gaps:
    Rc = Rc2(raw_occ) # calculate raw diversities (without taking gaps into account)
else:
    Rc = Rc1(raw_occ) # calculate raw diversities (without taking gaps into account)

m = 5
Bcs = m * Rc  # total number of pseudo-counts 


# For every column of the alignment:
if gaps:
    pseudo_occ = np.zeros((raw_occ.shape[0],21))
    for i in range(occ_wght.shape[0]):    
        Nc = occ_wght[i,:].sum()/1.0 # total number of counts in column i for weighted sequences
        Bc = Bcs[i] # total number of pseudo-counts in column i
    
        # For every residue:
        for j in range(len(AA)): 
            nca = occ_wght[i,j] # Nb of counts of aa j in column i
            bca_value = bca(Bc, AA[j], Nc, occ_wght[i,:], mtxfreq) # Nb of pseudo-counts of aa j in column i 
            pca = ( ( ( Nc/(Nc + Bc) ) * (nca/Nc) )+( ( Bc/(Nc + Bc) ) * (bca_value/Bc) ) )         
            pseudo_occ[i,j] = pca
else:
    pseudo_occ = np.zeros((raw_occ.shape[0],20))
    for i in range(occ_wght.shape[0]):    
        Nc = occ_wght[i,:-1].sum()/1.0 # total number of counts in column i for weighted sequences
#       print 'Nc: ', Nc
        Bc = Bcs[i] # total number of pseudo-counts in column i
    
        # For every residue:
        for j in range(len(AA)-1): 
            nca = occ_wght[i,j] # Nb of counts of aa j in column i
            bca_value = bca(Bc, AA[j], Nc, occ_wght[i,:-1], mtxfreq) # Nb of pseudo-counts of aa j in column i (we do not consider gaps)
            pca =( ( ( Nc/(Nc + Bc) ) * (nca/Nc) )  +( ( Bc/(Nc + Bc) ) * (bca_value/Bc) ) ) 
            pseudo_occ[i,j] = pca


## Print output ##
if gapc:
    for i in range(pseudo_occ.shape[0]):
        for j in range(pseudo_occ.shape[1]):
            print("%.4f"%pseudo_occ[i,j], end = " ")
        print("%.4f"%occ_wght[i,j+1], end = " ")
        print('')


else:
    for i in range(pseudo_occ.shape[0]):
#        summ=0
        for j in range(pseudo_occ.shape[1]):
            print("%.4f"%pseudo_occ[i,j], end = " ")
#           summ+=pseudo_occ[i,j]
#    print summ
        print('')



