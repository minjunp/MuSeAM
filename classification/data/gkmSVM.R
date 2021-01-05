#!/usr/bin/env Rscript

library(gkmSVM)
library(BSgenome.Mmusculus.UCSC.mm10)
library(IRanges)
library(GenomicRanges)
library(BSgenome.Mmusculus.UCSC.mm10.masked)

# input name
pos_bed="E13RACtrlF1_E13RAMutF1_DMR_toppos2000.bed"
# output names
pos_fa="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_pos.fa"
neg_fa="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_neg.fa"
neg_bed="E13RACtrlF1_E13RAMutF1_DMR_toppos2000_neg.bed"

genome=BSgenome.Mmusculus.UCSC.mm10.masked

genNullSeqs(pos_bed,nMaxTrials=20,xfold=1,genomeVersion='hg19', genome=genome, outputPosFastaFN=pos_fa, outputBedFN=neg_bed, outputNegFastaFN=neg_fa)
