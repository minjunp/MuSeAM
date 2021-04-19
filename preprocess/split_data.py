from .preprocess_data import preprocess
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
import numpy as np
import keras

def splitData(fasta_file, readout_file, taskType, partitionType):
    prep = preprocess(fasta_file, readout_file)
    dict = prep.one_hot_encode()

    fw_fasta = dict["forward"]
    rc_fasta = dict["reverse"]
    readout = dict["readout"]


    if partitionType == 'leaveOneOut':
        fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test = train_test_split(fw_fasta, rc_fasta, readout, test_size=0.1, shuffle=True)

    readout_train = np.array(readout_train)
    readout_test = np.array(readout_test)

    if taskType == 'classification':
        readout_train = keras.utils.to_categorical(readout_train, 2)
        readout_test = keras.utils.to_categorical(readout_test, 2)

    return fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test
