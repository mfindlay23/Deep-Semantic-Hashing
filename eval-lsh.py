import numpy as np
import sys
from itertools import chain
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy import spatial
import random

input_path = sys.argv[1]
hash_dict = {}
sim_dict = {}
precisions = []


def cosine_dist(s1, s2):
    s1 = list(s1)
    s2 = list(s2)
    s1 = np.asarray(s1, dtype='float64')
    s2 = np.asarray(s2, dtype='float64')
    assert len(s1) == len(s2)
    return 1 - spatial.distance.cosine(s1, s2)

def precision(top100, query_class):
    count = 0
    for _, c in top100:
        if c == query_class:
            count+=1
    a_precision = float(count)/100
    print ("Average precision, ", a_precision)
    return a_precision


with open(input_path, "r") as f:
    classes = set()
    for line in f.readlines():
        hash, c = line.split("]")
        c = c.split('.')[0]
        classes.add(c)
        hash_string = ''

        hash_string = tuple(hash[1:].split(','))
        hash_dict[hash_string] = c
        print("There are {0} classes").format(len(classes))

items=hash_dict.items()
random.shuffle(items)
count = 0
for query, query_class in items:
    top100 = []
    flag = False
    for key, key_class in items:
        if not flag:
            flag = True
            continue
        sim = cosine_dist(query, key)
        top100.append([sim, key_class])
        sorted(top100, key=lambda x: x[0])
        if len(top100) > 100:
            top100 = top100[:100]
    precisions.append(precision(top100, query_class))

avg_precisions = np.array(precisions)
mean_avg_precision = np.average(avg_precisions)


print("Mean average precision: ", mean_avg_precision)

with open("Mean_average_precision_result.txt", "w") as f:
    f.write("Mean average precision of RNN hasher: ")
    f.write(str(mean_avg_precision))
