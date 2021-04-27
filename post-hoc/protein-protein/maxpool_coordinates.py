import sys
import re
import os
import numpy as np

"""
FINAL_ver2: Put everything all together into one file then remove 0 outputs
"""
def remove_0_maxpool(dir='maxpool_final_control_removed'):
    ## get index for all filters
    directory = dir
    names = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
             name = os.path.join(directory, filename)
             names.append(name)
             continue
        else:
            continue
    #print(names)
    nums = []
    for i in range(len(names)):
        f = names[i]

        start = f.find('ter_',0,len(f))
        end = f.find('.txt',0,len(f))
        num = f[start+4:end]
        num = int(num)
        nums.append(num)

    # zip file to sort the dataset
    zipped = zip(nums, names)
    zipped = sorted(zipped)

    nums, files = zip(*zipped)

    names = files

    for i in range(len(names)):
        with open(names[i]) as f:
            seps = []
            vals = [line.rstrip() for line in f]
            for j in range(len(vals)):
                sep = vals[j].split(" ")
                #print(sep[0])
                if sep[0] != '0.0':
                    seps.append(sep)

            #print(len(seps))
        print(i)

        text_file = open("./maxpool_final2_control_removed/filter_%d"%i + ".txt", "w")
        for line in range(len(seps)):
            text_file.write(seps[line][1]+ ' ' + seps[line][2]+ ' ' +
            seps[line][3] + '\n')
        text_file.close()


"""
FINAL_ver1: Put everything all together into one file
"""
def comb_peak_pool(peak_file='peaks.txt', ind_dir='maxpool_index_ver2_control_removed', max_dir='maxpool_filter_control_removed'):
    with open(peak_file) as f:
        peaks = [line.rstrip() for line in f]
    peaks = np.reshape(peaks, (2236,1))

    ## get index for all filters
    directory = ind_dir
    names = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
             name = os.path.join(directory, filename)
             names.append(name)
             continue
        else:
            continue

    ## get maxpool outputs for all filters
    directory = max_dir
    filter_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
             name = os.path.join(directory, filename)
             filter_names.append(name)
             continue
        else:
            continue

    for i in range(len(names)):
        index_f = names[i]
        maxpool_f = filter_names[i]
        index_pos = np.loadtxt(index_f)
        maxpool_vals = np.loadtxt(maxpool_f)
        index_pos = np.reshape(index_pos, (2236,2))
        maxpool_vals = np.reshape(maxpool_vals, (2236,1))
        concat_everything = np.concatenate((maxpool_vals, peaks,index_pos), axis=1)

        #with open("./maxpool_final/filter_%d"%i + ".txt", "w") as text_file:
        #    text_file.write(concat_everything)

        #np.savetxt("./maxpool_final/filter_%d"%i + ".txt")
        #np.savetxt(os.path.join("./maxpool_final/filter_%d"%i + ".txt"), concat_everything)
        text_file = open("./maxpool_final_control_removed/filter_%d"%i + ".txt", "w")

        """
        We have peak information for all sequences in order
        """
        for line in range(len(concat_everything)):
            text_file.write(concat_everything[line][0]+ ' ' + concat_everything[line][1]+ ' ' +
            concat_everything[line][2] + ' ' + concat_everything[line][3] + '\n')
        text_file.close()



"""
Preprocessing for files in maxpool_index folder,
which is obetained from the model.py maxpool_coordinates function
"""


"""
We have max pooling outputs in order of sequences for each filter
"""
## I'm trying to save file for each filter level
def reshape_maxpool_output(dir = 'max_outputs'):
    current_path = os.getcwd() # current directory

    if dir == 'max_outputs_control_removed':
        names = []
        for filename in os.listdir(dir):
            if filename.startswith("maxfile"):
                 name = os.path.join(dir, filename)
                 names.append(name)
                 continue
            else:
                continue
        names = sorted(names)

        res = np.empty([512, 1])
        for i in range(len(names)):
            f = np.loadtxt(current_path+'/'+names[i])
            f = np.reshape(f, (512, 1))
            res = np.concatenate((res,f), axis=1)
            #res = np.vstack((res,f))
        for i in range(512):
            index = i
            if index < 10:
                index = '00'+str(index)
            elif index >= 10 and index < 100:
                index = '0'+str(index)
            elif index >= 100:
                index = str(index)
            np.savetxt(os.path.join('./maxpool_filter_control_removed/filter_%s'%index+'.txt'), res[i,1:]) ## save it so that each filter has 2440 maxpool outputs

    if dir == 'shuffled_max_outputs':
        file_names = []
        for filename in os.listdir(dir):

            name = os.path.join(dir, filename)
            file_names.append(name)
        file_names = sorted(file_names)

        for i in file_names:
            str_len = len(i)
            start = i.find('outputs/',0,str_len)

            end = str_len
            new_name = i[start+8:end]


            path = current_path+'/shuffled_maxpool_filter/'+new_name
            os.mkdir(path)

            names = []
            for filename in os.listdir(current_path+'/shuffled_max_outputs/'+new_name):
                if filename.startswith("maxfile"):
                     names.append(filename)
                     continue
                else:
                    continue
            names = sorted(names)

            res = np.empty([512, 1])
            for i in range(len(names)):
                f = np.loadtxt(current_path+'/'+'shuffled_max_outputs/'+new_name+'/'+names[i])
                f = np.reshape(f, (512, 1))
                res = np.concatenate((res,f), axis=1)
                #res = np.vstack((res,f))
            for i in range(512):
                index = i
                if index < 10:
                    index = '00'+str(index)
                elif index >= 10 and index < 100:
                    index = '0'+str(index)
                elif index >= 100:
                    index = str(index)
                np.savetxt(os.path.join('./shuffled_maxpool_filter/'+new_name+'/filter_%s'%index+'.txt'), res[i,1:]) ## save it so that each filter has 2440 maxpool outputs

#write_peak() # get peaks.txt by calling fasta file
maxpool_index_ver2() # need to get index from model.py
sys.exit()
#reshape_maxpool_output(dir = 'shuffled_max_outputs') # reshapes max pool outputs that we get from model.py in entire model function
#reshape_maxpool_output(dir = 'max_outputs_control_removed')

comb_peak_pool() #final version 1 which combines peak file with maxpool file

remove_0_maxpool() #final version 2
