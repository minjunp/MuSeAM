import sys
import pandas as pd
import numpy as np
from glob import glob

hepg2_file = pd.read_csv('HepG2.Genc.gene_name', sep=' ', header=None)
hepg2_expressed_tfs = hepg2_file[2].str.upper().tolist()

tomtom_dir_path = '../tomtom_outputs_evalue_5_v2/result*/tomtom.tsv'
tomtom_dir = sorted(glob(tomtom_dir_path))

tomtoms = []
with open('hepg2_expressed_list.txt', 'w') as file:
    for i in tomtom_dir:
        motif_num = i[-14:-11]
        motif_name = 'motif_'+motif_num

        fout = pd.read_csv(i, header=None, sep='\t')

        # if starting with '#', it means no match
        if fout.iloc[1,0].startswith('#'):
            tomtoms.append('Novel_Motif')
        else:
            fout = fout[:-3]
            TFs = fout[1][1:]
            tf_names = []
            for tf in TFs:
                tf_name = tf.split('_')[0]

                # see if they are expressed in hepg2
                if tf_name in hepg2_expressed_tfs:
                    tf_names.append(tf_name)

            file.write(f'{motif_name},{tf_names}\n')
