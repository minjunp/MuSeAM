import numpy as np
from heapq import nlargest, nsmallest
import sys
import os

vals = np.loadtxt('dense_weights.txt')

largest = nlargest(137, vals)
smallest = nsmallest(128, vals)

#largest_ind = vals.index(largest)
#smallest_ind = vals.index(smallest)

#print(largest_ind)

########################################
dense_weights_ordered = np.loadtxt("dense_weights.txt")
dense_weights_ordered_indices = np.argsort(dense_weights_ordered)

print("indices of 128 repressors: ")
repressors_indices = dense_weights_ordered_indices[0:128]
print(repressors_indices)
print("\nindices of 137 activators: ")
activators_indices = dense_weights_ordered_indices[-138:-1]
print(activators_indices)
print("\n new activators indices")
print(np.flip(activators_indices))


print("\n \n")

print("the values of repressors: ")
repressors_values = dense_weights_ordered[repressors_indices[0:128]]
print(repressors_values)
print("\nthe values of activators: ")
activators_values = dense_weights_ordered[np.flip(activators_indices)[0:137]]
#activators_values = np.sort(activators_values)[::-1]
print(activators_values)


"""
Get max pool filters from maxpool_filter folder
"""
# repressors_indices
# activators_indices

directory = '/Users/minjunp/Documents/research/baylor/mpra_project/liver_enhancer/maxpool_filter'
names = []
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
         name = os.path.join(directory, filename)
         names.append(name)
         continue
    else:
        continue

nums = []
for i in range(len(names)):
    f = names[i]

    start = f.find('/filter',0,len(f))
    end = f.find('.txt',0,len(f))
    num = f[start+8:end]
    num = int(num)

    load_file = np.loadtxt(f)

    if num in activators_indices:
        np.savetxt(os.path.join('./activators_137', 'activator_%d'%i+'.txt'), load_file)

    if num in repressors_indices:
        np.savetxt(os.path.join('./repressors_128', 'repressor_%d'%i+'.txt'), load_file)
