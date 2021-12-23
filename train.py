from main import nn_model
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
    dict["epochs"] = int(dict["epochs"])
    dict["batch_size"] = int(dict["batch_size"])
    dict["alpha"] = int(dict["alpha"])
    dict["beta"] = float(dict["beta"])

    return dict

def main(argv = None):
    if argv is None:
        argv = sys.argv
        #e.g. sequences.fa
        fasta_file = argv[1]
        #e.g. wt_readout.dat
        readout_file = argv[2]
        #e.g. parameter1.txt
        parameter_file = argv[3]

    ## excute the code
    start_time = time.time()
    dict = train(parameter_file)

    nn_model(fasta_file, readout_file, filters=dict["filters"], kernel_size=dict["kernel_size"],
            epochs=dict["epochs"], batch_size=dict["batch_size"], alpha=dict["alpha"], beta=dict["beta"])

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    sys.exit(main())