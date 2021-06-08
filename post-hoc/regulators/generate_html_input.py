import sys
import numpy as np
import pandas as pd
from glob import glob

def process_data(fwd_logos, rc_logos, rankfile, tomtom_dir_path, output_name):
    fwd_files = sorted(glob(fwd_logos))
    rc_files = sorted(glob(rc_logos))
    d = {'fwd': fwd_files, 'rc': rc_files}
    df = pd.DataFrame(d)

    def find_between(s, first, last):
        try:
            start = s.index(first) + len(first)
            end = s.index(last,start)
            return s[start:end]
        except ValueError:
            return ""
    nums = []
    for f in fwd_files:
        num = find_between(f, "logos/fwd_", ".png" )
        nums.append(num)

    dense_weights = np.loadtxt('dense_weights.txt')

    ranks = np.loadtxt(rankfile)
    # argsort one more time to get the ranks of order
    ranks = np.argsort(ranks)

    ## now I need to get tomtom matches by looking at .tsv files
    ## I also need to get the FC weights for each filters
    tomtom_dir = sorted(glob(tomtom_dir_path))
    #df['tomtom'] = tomtom_dir

    tomtoms = []
    for i in tomtom_dir:
        fout = pd.read_csv(i, header=None, sep='\t')
        #fout = pd.read_csv('/Users/minjunp/Documents/baylor/methylation/outs/motif_analysis/motif_files/tomtom_outputs/result_7/tomtom.tsv',header=None, sep='\t')
        if fout.iloc[1,0].startswith('#'):
            tomtoms.append('Novel_Motif')
        else:
            tomtoms.append(fout.iloc[1,1])

    df['tomtom'] = tomtoms
    df['ranks'] = ranks
    df['filter_num'] = nums
    df['dense_weight'] = dense_weights
    df = df.sort_values(by=['ranks'], ascending=False)
    df['ranks'] = df['ranks'].values[::-1]
    # save to .csv file
    df.to_csv(output_name, sep=' ', index=False, header=None)

fwd_logos = '../../saved_model/MuSeAM_regression/motif_files/motif_files/logos/fwd*.png'
rc_logos = '../../saved_model/MuSeAM_regression/motif_files/motif_files/logos/rc*.png'
rankfile = 'motif_rank.txt'
tomtom_dir_path = '../../saved_model/MuSeAM_regression/motif_files/motif_files/tomtom_overlap_5_outputs/result*/tomtom.tsv'
output_name = 'html_input_overlap_5.txt'

#tomtom_dir_path = '../../saved_model/MuSeAM_regression/motif_files/motif_files/tomtom_outputs/result*/tomtom.tsv'
#output_name = 'html_input.txt'
process_data(fwd_logos, rc_logos, rankfile, tomtom_dir_path, output_name)

# fwd_logos = '../../saved_model/MuSeAM_regression_split/motif_files/motif_files/logos/fwd*.png'
# rc_logos = '../../saved_model/MuSeAM_regression_split/motif_files/motif_files/logos/rc*.png'
# rankfile = 'motif_split_rank.txt'
# tomtom_dir_path = '../../saved_model/MuSeAM_regression_split/motif_files/motif_files/tomtom_outputs/result*/tomtom.tsv'
# output_name = 'html_input_split.txt'
# process_data(fwd_logos, rc_logos, rankfile, tomtom_dir_path, output_name)
