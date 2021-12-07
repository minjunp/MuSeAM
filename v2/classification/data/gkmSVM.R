#!/usr/bin/env Rscript

library(gkmSVM)
library(BSgenome.Mmusculus.UCSC.mm10)
library(IRanges)
library(GenomicRanges)
library(BSgenome.Mmusculus.UCSC.mm10.masked)

fname <- read.csv('config.csv')

# input name
#pos_bed="E13RACtrlF1_E13RAMutF1_DMR_toppos2000.bed"
pos_bed <- fname$pos_bed
# output names
pos_fa <- fname$pos_fa
neg_fa <- fname$neg_fa
neg_bed <- fname$neg_bed
#pos_fa="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_pos.fa"
#neg_fa="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_neg.fa"
#neg_bed="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_neg.bed"

if (fname$organism == 'hg19'){
genNullSeqs(pos_bed,nMaxTrials=20,xfold=1,genomeVersion='hg19', outputPosFastaFN=pos_fa, outputBedFN=neg_bed, outputNegFastaFN=neg_fa)
}

if (fname$organism == 'mm10'){
genome=BSgenome.Mmusculus.UCSC.mm10.masked

genNullSeqs(pos_bed,nMaxTrials=20,xfold=1,genomeVersion='hg19', genome=genome, outputPosFastaFN=pos_fa, outputBedFN=neg_bed, outputNegFastaFN=neg_fa)
}
