import os
import pandas as pd
from gimmemotifs.scanner import Scanner
from gimmemotifs.motif import read_motifs
from gimmemotifs.config import MotifConfig
from tangerine.preprocess.utils import peak2fasta, scan_dna_for_motifs
from tangerine.preprocess.logger import logger

# Copied from previous version of gimme motifs repository.
# Somehow this has been removed in the latest update.
def default_motifs():
    """Return list of Motif instances from default motif database."""
    config = MotifConfig()
    d = config.get_motif_dir()
    m = config.get_default_params()["motif_db"]

    if not d or not m:
        raise ValueError("default motif database not configured")

    fname = os.path.join(d, m)
    with open(fname) as f:
        motifs = read_motifs(f)

    return motifs


class ScannerMotifs(object):
    """
    Class to scan motifs in specified region of genome.
    Acts as a wrapper around GimmeMotifs to extract TF binding site priors.
    """
    def __init__(self, genome, n_cpus=1, fpr=0.02, scan_width=5000) -> None:
        """Constructor for ScannerMotifs class

        Args:
            genome (str): Genome which will be used for scanning (currently supports mm10 and hg19)
            n_cpus (int, optional): Number of CPU cores to use. Defaults to 1.
            fpr (float, optional): False positive rate. Defaults to 0.02.
            scan_width (int, optional): Width of genomic region upstream of promoter to scan (in bp). Defaults to 5000.
        """
        self.genome = genome
        
        logger.info(f"Initializing ScannerMotifs for genome: {genome} with scan width: {scan_width}bp")

        self.scanner_object = Scanner(ncpus=n_cpus)
        self.motifs = default_motifs()

        self.scanner_object.set_background(genome=genome)
        self.scanner_object.set_motifs(self.motifs)
        self.scanner_object.set_threshold(fpr=fpr)

        self.scan_width = scan_width
        self.all_tfs = self.get_all_motifs()

    def get_all_motifs(self):
        """Makes a consolidated set of TFs in the database

        Returns:
            list: List of consolidated TFs in the database
        """
        all_tfs = []

        for motif in self.motifs:
            all_tfs.extend(motif.factors['direct'])
            all_tfs.extend(motif.factors['indirect\nor predicted'])

        all_tfs = [a.capitalize() for a in all_tfs]
        all_tfs = list(set(all_tfs))
        return all_tfs

    def scan_motifs_single(self, chr_name, start, strand):
        """Scan for TF motifs upstream of a single gene

        Args:
            chr_name (str): Chromosome name
            start (int): start position in bp
            strand (int): strandedness of the gene  

        Returns:
            pandas DataFrame: Results of scan in dataframe
        """
        peakstr = f'{chr_name}_{start - (self.scan_width * strand)}_{start}'
        target_sequences = peak2fasta(peak_ids=[peakstr], ref_genome=self.genome)
        scanned_df = scan_dna_for_motifs(scanner_object=self.scanner_object, motifs_object=self.motifs, 
                                      sequence_object=target_sequences, divide=100000)
        
        return scanned_df
    
    def get_filtered_tfs(self, scanned_df, score=None, include_indirect=False):
        """Filters the scanned dataframe to extract valid transcription factors.
        """
        if score:
            assert score > 0, "Invalid score, cannot filter binding sites"
            scanned_df = scanned_df[scanned_df.score > score]
        
        temp_tf_list = list(scanned_df.factors_direct.unique())
        if include_indirect:
            temp_tf_list += list(scanned_df.factors_indirect.unique())

        tf_list = []

        for element in temp_tf_list:
            temp = element.split(',')
            for e in temp:
                if len(e.strip()) > 0:
                    tf_list.append(e.strip().capitalize())

        tf_list = list(set(tf_list))
        return tf_list
    
    def run_single_scan(self, chr_name, start, strand, score=None, include_indirect=False):
        """Run single scan and return list of TFs found

        Args:
            chr_name (str): Chromosome name
            start (int): start position
            strand (int): strandedness (-1 or 1)
            score (float, optional): Threshold score to filter. Defaults to None.
            include_indirect (bool, optional): Whether or not to include indirect TF binding sites. Defaults to False.

        Returns:
            list: List of TFs whose motif was found in the region
        """
        scanned_df = self.scan_motifs_single(chr_name, start, strand)
        tf_list = self.get_filtered_tfs(scanned_df, score, include_indirect)
        return tf_list
