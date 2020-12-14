from model import nn_model
#from one_seq_pipeline import nn_model
#from one_seq_sample import nn_model
import time
import sys

# get dictionary from text file
def train(file_name):
    dict = {}
    with open(file_name) as f:
        for line in f:
           (key, val) = line.split()
           dict[key] = val

    # change string values to integer values
    dict["filters"] = int(dict["filters"])
    dict["kernel_size"] = int(dict["kernel_size"])
    dict["pool"] = int(dict["pool"])
    dict["epochs"] = int(dict["epochs"])
    dict["batch_size"] = int(dict["batch_size"])

    return dict

def main(argv = None):
    if argv is None:
        argv = sys.argv

        #input args
        fasta_file = argv[1]
        #e.g. sequences.fa
        readout_file = argv[2]
        #e.g. wt_readout.dat
        parameter_file = argv[3]
        #e.g. parameter1.txt

    ## excute the code
    start_time = time.time()

    dict = train(parameter_file)

    model = nn_model(fasta_file, readout_file, filters=dict["filters"], kernel_size=dict["kernel_size"], pool_type=dict["pool_type"], pool=dict["pool"], regularizer=dict["regularizer"],
            activation_type=dict["activation_type"], epochs=dict["epochs"], batch_size=dict["batch_size"])

    #model.eval()
    model.cross_val_custom()

    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    sys.exit(main())

#train('parameter1.txt')
