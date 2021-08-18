import sys
import numpy as np
import pandas as pd
from glob import glob

def process_data(fwd_logos, rc_logos, rankfile, tomtom_dir_path, output_name, dense_weights_file):
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

    dense_weights = np.loadtxt(dense_weights_file)

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

fwd_logos = '/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_liver_enhancer/motif_files/motif_files/logos/fwd*.png'
rc_logos = '/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_liver_enhancer/motif_files/motif_files/logos/rc*.png'
rankfile = '/Users/minjunpark/Documents/MuSeAM/post-hoc/regulators/generate_html/liver_enhancer/motif_rank.txt'
tomtom_dir_path = '/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_liver_enhancer/motif_files/motif_files/tomtom_outputs_evalue_10/result*/tomtom.tsv'
output_name = 'html_input_evalue_10_liver_enhancer.txt'
dense_weights_file = '/Users/minjunpark/Documents/MuSeAM/post-hoc/regulators/generate_html/liver_enhancer/dense_weights.txt'
process_data(fwd_logos, rc_logos, rankfile, tomtom_dir_path, output_name, dense_weights_file)

# fwd_logos = '/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_silencer/motif_files/motif_files/logos/fwd*.png'
# rc_logos = '/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_silencer/motif_files/motif_files/logos/rc*.png'
# rankfile = '/Users/minjunpark/Documents/MuSeAM/post-hoc/regulators/generate_html/silencer/motif_silencer_rank.txt'
# tomtom_dir_path = '/Users/minjunpark/Documents/MuSeAM/saved_model/MuSeAM_regression_silencer/motif_files/motif_files/tomtom_outputs_evalue_10/result*/tomtom.tsv'
# output_name = 'html_input_evalue_10_silencer.txt'
# dense_weights_file = '/Users/minjunpark/Documents/MuSeAM/post-hoc/regulators/generate_html/silencer/dense_weights_silencer.txt'
# process_data(fwd_logos, rc_logos, rankfile, tomtom_dir_path, output_name, dense_weights_file)
