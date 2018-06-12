from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import seq2seq
import utils
import sklearn
import numpy as np
import sys
from itertools import chain
import time

def hamming_dist(s1, s2):
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def precision(top100, query_class):
    count = 0
    for _, c in top100:
        if c == query_class:
            count+=1
    a_precision = float(count)/100
    print ("Average precision, ", a_precision)
    return a_precision

if __name__ == "__main__":

    print("\n\nLoading pre-trained model weights...")
    time.sleep(6)
    data_path = '20news-bydate-train'
    utils.DataUtils.get_20news_dataset(data_path)

    batch_size = 64  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000  # Number of samples to train on.
    # Path to the data txt file on disk.

    encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens = utils.DataUtils.prep_train_data(0, 1000)
    seq_model = seq2seq.seq2seq(latent_dim, num_encoder_tokens, num_decoder_tokens)
    seq_model.load_existing()

    #enc_test, _, _, _, _= utils.DataUtils.prep_train_data(1000, 1100)
    for seq_index in range(1):
        print("\nSelecting document...")
        time.sleep(5)
        print("\nHashing document\n")
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        states_value = np.array(seq_model.hash(input_seq))
        hash_code = utils.hash(states_value)
        hash_class = utils.DataUtils.Y_train[seq_index].split(".")[0].strip(",")
        h = ''
        for val in hash_code:
            h+=str(val)
        print("Hash value of document: \n", h)
        print ("\nClass of hashed document: ", hash_class)
        print("\n")

    time.sleep(5)

    input_path = "hashed_documents/all_hash_values.txt"
    hash_dict = {}
    sim_dict = {}
    precisions = []
    print("Getting hash values of other documents...\n")
    time.sleep(5)

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

    query = h
    query_class = hash_class
    top100 = []
    print("\nComparing hash values, finding 100 most similar\n")
    for key, key_class in hash_dict.items():
        sim = hamming_dist(query, key)
        top100.append([sim, key_class])
    top100 = sorted(top100, key=lambda x: x[0], reverse=True)

    print("\nClasses of most relevant documents:\n")
    for i, v in enumerate(top100):
        print(str(i) + ". " + "sim: " + str(v[0]) + " class: " + str(v[1]))
        if (i == 100):
            break
    precisions.append(precision(top100[:100], query_class))

    avg_precisions = np.array(precisions)
