'''
These functions are copied and adapted from CellOracle (https://github.com/morris-lab/CellOracle)

Copyright 2020 Kenji Kamimoto, Christy Hoffmann, Samantha Morris

The software is provided under a modified Apache License Version 2.0.
The software may be used under the Apache License below for NON-COMMERCIAL ACADEMIC PURPOSES only.
For any other use of the Work, including commercial use, please contact Morris lab.
'''

import pandas as pd
import numpy as np
from gimmemotifs.fasta import Fasta
from genomepy import Genome
from gimmemotifs.config import DIRECT_NAME, INDIRECT_NAME

def peak_M1(peak_id):
    """
    Take a peak_id (index of bed file) as input,
    then subtract 1 from Start position.

    Args:
        peak_id (str): Index of bed file. It should be made of "chromosome name", "start position", "end position"
            e.g. "chr11_123445555_123445577"
    Returns:
        str: Processed peak_id.

    Examples:
        >>> a = "chr11_123445555_123445577"
        >>> peak_M1(a)
        "chr11_123445554_123445577"
    """
    chr_, start, end = decompose_chrstr(peak_id)
    return chr_ + "_" + str(int(start)-1) + "_" + end


def list2str(li):
    """
    Convert list of str into merged one str with " ,".
    See example below for detail.

    Args:
        li (list of str): list

    Returns:
        str: merged str.

    Examples:
        >>> a = ["a", "b", "c"]
        >>> list2str(a)
        "a, b, c"
    """
    return ", ".join(li)


def scan_dna_for_motifs(scanner_object, motifs_object, sequence_object, divide=100000, verbose=True):
    '''
    This is a wrapper function to scan DNA sequences searchig for Gene motifs.

    Args:

        scanner_object (gimmemotifs.scanner): Object that do motif scan.

        motifs_object (gimmemotifs.motifs): Object that stores motif data.

        sequence_object (gimmemotifs.fasta): Object that stores sequence data.

    Returns:
        pandas.dataframe: scan results is stored in data frame.

    '''

    li  = []
    # If the gimmemotifs version is larger than 0.17.0, it will automatically return the progress bar,
    # So we don't need to make tqdm qbar here.

    iter = scanner_object.scan(sequence_object)

    for i, result in enumerate(iter):
        seqname = sequence_object.ids[i]
        for m,matches in enumerate(result):
            motif = motifs_object[m]
            for score, pos, strand in matches:
                li.append(np.array([seqname,
                                    motif.id,
                                    list2str(motif.factors[DIRECT_NAME]),
                                    list2str(motif.factors[INDIRECT_NAME]),
                                    score, pos, strand]))

    if len(li)==0:
        df = pd.DataFrame(columns=["seqname",
                               "motif_id",
                               "factors_direct",
                               "factors_indirect",
                               "score", "pos", "strand"])
    else:

        remaining = 1
        LI = []
        k = 0
        while remaining == 1:
            #print(k)
            tmp_li = li[divide*k:min(len(li), divide*(k+1))]

            tmp_li = np.stack(tmp_li)
            df = pd.DataFrame(tmp_li,
                              columns=["seqname",
                                       "motif_id",
                                       "factors_direct",
                                       "factors_indirect",
                                       "score", "pos", "strand"])
            df.score = df.score.astype(float)
            df.pos = df.pos.astype(int)
            df.strand = df.strand.astype(int)

            # keep peak id name
            df.seqname = list(map(peak_M1, df.seqname.values))

            LI.append(df)


            if divide*(k+1) >= len(li):
                remaining = 0
            k += 1

        df = pd.concat(LI, axis=0)
    return df


def decompose_chrstr(peak_str):
    """
    Take peak name as input and return splitted strs.

    Args:
        peak_str (str): peak name.

    Returns:
        tuple: splitted peak name.

    Examples:
       >>> decompose_chrstr("chr1_111111_222222")
       "chr1", "111111", "222222"
    """
    *chr_, start, end = peak_str.split("_")
    chr_ = "_".join(chr_)

    return chr_, start, end


def peak2fasta(peak_ids, ref_genome, genomes_dir):

    '''
    Convert peak_id into fasta object.

    Args:
        peak_id (str or list of str): Peak_id.  e.g. "chr5_0930303_9499409"
            or it can be a list of peak_id.  e.g. ["chr5_0930303_9499409", "chr11_123445555_123445577"]

        ref_genome (str): Reference genome name.   e.g. "mm9", "mm10", "hg19" etc

        genomes_dir (str): Installation directory of Genomepy reference genome data.

    Returns:
        gimmemotifs fasta object: DNA sequence in fasta format

    '''
    genome_data = Genome(ref_genome, genomes_dir=genomes_dir)

    def peak2seq(peak_id):
        chromosome_name, start, end = decompose_chrstr(peak_id)
        locus = (int(start),int(end))

        tmp = genome_data[chromosome_name][locus[0]:locus[1]]
        name = f"{tmp.name}_{tmp.start}_{tmp.end}"
        seq = tmp.seq
        return (name, seq)


    if type(peak_ids) is str:
        peak_ids = [peak_ids]

    fasta = Fasta()
    for peak_id in peak_ids:
        name, seq = peak2seq(peak_id)
        fasta.add(name, seq)

    return fasta
