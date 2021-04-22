from .preprocess_data import preprocess
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
import numpy as np
import keras

def sharpr():
    # Load Train, Test, and Valid data
    train_seq = np.load('./data/sharpr/train_seqs.npy')
    train_readout = np.load('./data/sharpr/train_readout.npy')
    fwd_train = train_seq[::2, :, :]
    rc_train = train_seq[1::2, :, :]
    readout_train = train_readout[::2, :]

    fwd_test = np.load('./data/sharpr/test_seqs_fwd.npy')
    rc_test = np.load('./data/sharpr/test_seqs_rc.npy')
    readout_test = np.load('./data/sharpr/test_readout.npy')

    valid_seq = np.load('./data/sharpr/valid_seqs.npy')
    valid_readout = np.load('./data/sharpr/valid_readout.npy')
    fwd_valid = valid_seq[::2, :, :]
    rc_valid = valid_seq[1::2, :, :]
    readout_valid = valid_readout[::2, :]

    return fwd_train, rc_train, readout_train, fwd_test, rc_test, readout_test,fwd_valid, rc_valid, readout_valid

def splitData(fasta_file, readout_file, partitionType = None, taskType = None):
    prep = preprocess(fasta_file, readout_file)
    dict = prep.one_hot_encode()

    fwd_fasta = dict["forward"]
    rc_fasta = dict["reverse"]
    readout = dict["readout"]

    if taskType == 'binary_classification':
        readout = keras.utils.to_categorical(readout, 2)

    if partitionType == '10Fold' or 'fitAll':
        readout = np.array(readout)
        return fwd_fasta, rc_fasta, readout

    if partitionType == 'leaveOneOut':
        fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test = train_test_split(fwd_fasta, rc_fasta, readout, test_size=0.1, shuffle=True)
        readout_train = np.array(readout_train)
        readout_test = np.array(readout_test)

    return fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test
