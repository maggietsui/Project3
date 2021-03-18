def read_train_seqs(pos_file, neg_file):
    """
    Reads in positive and negative sequences from paths
    into two lists, pos and neg. For negative sequences,
    skips the lines starting with ">"

    Parameters
    ---------
    pos_file
        path to positive examples
    neg_file
        path to negative examples
        
    Returns
    ---------
    two lists, where each element is a sequence from the file
    """
    with open(pos_file) as f:
        pos = f.read().splitlines()
    
    neg = []
    seq = ''
    for line in open(neg_file):
        if line.startswith(">"):
            if seq != '':
                neg.append(seq)
                seq = ''
        else:
            seq += line.strip() 
    neg.append(seq)
    return pos,neg

def encode_seq(sequence):
    """
    Performs one-hot encoding of a nucleotide sequence,
    where each nucleotide is represented by a binary vector
    of length 4, ie: [1, 0, 0, 0], where a 1 corresponds to
    which nucleotide it is: [A, C, G, T]

    Parameters
    ---------
    sequence
        the sequence string to encode
        
    Returns
    ---------
    A one-hot encoded sequence represented as a list of lists,
    each with length 4
    """
    encoded = []
    for nuc in sequence:
        if nuc == 'A':
            encoded+=[1,0,0,0]
        elif nuc == 'C':
            encoded+=[0,1,0,0]
        elif nuc == 'G':
            encoded+=[0,0,1,0]
        elif nuc == 'T':
            encoded+=[0,0,0,1]
    return encoded
