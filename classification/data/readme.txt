# Steps for MuSeAM data processing
1. Prepare Bedfile as an input
2. Run "Rscript gkmSVM.R" to get positive & negative fasta files
3. Run "python process_fasta.py" to re-format to write all sequences in one line
4. Generate readout file and concatenate positive and negative sequences
5. Run MuSeAM model!


