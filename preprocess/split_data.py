from .preprocess_data import preprocess
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
import numpy as np
import keras

def splitData(fasta_file, readout_file, partitionType = None, taskType = None):
    prep = preprocess(fasta_file, readout_file)
    dict = prep.one_hot_encode()

    fwd_fasta = dict["forward"]
    rc_fasta = dict["reverse"]
    readout = dict["readout"]

    if taskType == 'binary_classification':
        readout = keras.utils.to_categorical(readout, 2)

    if partitionType == '10Fold':
        readout = np.array(readout)
        return fwd_fasta, rc_fasta, readout

    if partitionType == 'leaveOneOut':
        fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test = train_test_split(fwd_fasta, rc_fasta, readout, test_size=0.1, shuffle=True)
        readout_train = np.array(readout_train)
        readout_test = np.array(readout_test)

    return fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test
