import numpy as np
import sys
from itertools import chain
from sklearn.metrics import average_precision_score

input_path = sys.argv[1]
hash_dict = {}
sim_dict = {}
precisions = []

def hamming_dist(s1, s2):
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def precision(top100, query_class, classes):
    count = 0
    for _, c in top100:
        if c == query_class:
            print (c)
            print (query_class)
            count+=1
    a_precision = float(count)/100
    print ("Average precision, ", a_precision)
    return a_precision


with open(input_path, "r") as f:
    classes = {}
    for line in f.readlines():
        hash, c = line.split("]")
        c = c.split(".")[0].strip(",")
        if c in classes:
            classes[c]+=1
        else:
            classes[c] = 1
        hash_string = ''
        for val in hash:
            if val == '1' or val == '0':
                hash_string+=val

        hash_dict[hash_string] = c

for query, query_class in hash_dict.items():
    top100 = []
    for key, key_class in hash_dict.items():
        sim = hamming_dist(query, key)
        top100.append([sim, key_class])
    top100 = sorted(top100, key=lambda x: x[0], reverse=True)
    if len(top100) > 100:
        top100 = top100[:100]
    precisions.append(precision(top100, query_class, classes))

avg_precisions = np.array(precisions)

print("Mean average precision: ", np.average(avg_precisions))

with open("Mean_average_precision_result.txt", "w") as f:
    f.write("Mean average precision of RNN hasher: ")
    f.write(str(np.average(avg_precisions)))
