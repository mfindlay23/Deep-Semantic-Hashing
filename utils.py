import os
import scipy
import tqdm
import numpy as np
"""
def getSession(gpu_num, gpu_fraction):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

    num_threads = os.environ.get("OMP_NUM_THREADS")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)
"""
def hash(states):
    state_string = states[0]+states[1]
    return [1 if i >0 else 0 for i in state_string]

class DataUtils(object):
    #X train contains vector of words representations
    #Y train contains labels
    #Files is the name of the file info originated from
    #Ordered by index
    #Converted to numpy arrays after calling
    X_train = []
    Y_train = []
    files = []
    word_vec_train = []

    @staticmethod
    def get_20news_dataset(file_path):
        for dname in os.listdir(file_path):
            if(dname != ".DS_Store"):
                dname = os.path.join(file_path, dname)
                for fname in os.listdir(dname):
                    fname = os.path.join(dname, fname)
                    try:
                        with open(fname, "rb") as f:
                            file_array = []
                            DataUtils.files.append(fname)
                            for line in f.readlines():
                                for word in line.split():
                                    file_array.append(word)
                            DataUtils.X_train.append(file_array)
                            DataUtils.Y_train.append(dname.split("/")[1])
                    except FileNotFoundError:
                        print ("Did not find file: ", fname)

        DataUtils.X_train = np.array(DataUtils.X_train)
        DataUtils.Y_train = np.array(DataUtils.Y_train)
        DataUtils.files = np.array(DataUtils.files)

    @staticmethod
    def get_vector_embeddings():
        # Load Google's pre-trained Word2Vec model.
        model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        #Create word2vec matrix
        for doc in utils.DataUtils.X_train:
            doc_vec = []
            for word in doc:
                try:
                    try:
                        doc_vec.append(model.get_vector(word.decode('utf-8')))
                    except UnicodeDecodeError:
                        #ignore broken embeddings
                        pass
                except KeyError:
                    print("Word not in vocab, skipping")

        DataUtils.word_vec_train.append(doc_vec)
