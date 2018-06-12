import os
import scipy
import tqdm
import numpy as np

def hash(states):
    state_string = states.flatten()
    return [1 if i >0 else 0 for i in state_string]

def map(file_path):
    with open(file_path, "r") as f:
        for line in f.readlines():
            pass


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

        from sklearn.utils import shuffle
        DataUtils.X_train, DataUtils.Y_train = shuffle(DataUtils.X_train, DataUtils.Y_train, random_state=42)

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

    @staticmethod
    def prep_train_data(start_ind=0, end_ind=100):
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        for doc in DataUtils.X_train[start_ind:end_ind]:
            for words in doc:
                try:
                    w = words.decode('utf-8')
                    input_text = w
                    target_text = w
                    # We use "tab" as the "start sequence" character
                    # for the targets, and "\n" as "end sequence" character.
                    target_text = '\t' + target_text + '\n'
                    input_texts.append(input_text)
                    target_texts.append(target_text)
                    for char in input_text:
                        if char not in input_characters:
                            input_characters.add(char)
                    for char in target_text:
                        if char not in target_characters:
                            target_characters.add(char)
                except UnicodeDecodeError:
                    pass

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        input_token_index = dict(
                             [(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict(
                              [(char, i) for i, char in enumerate(target_characters)])

        encoder_input_data = np.zeros(
                                  (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
                                  dtype='float32')
        decoder_input_data = np.zeros(
                                  (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
                                  dtype='float32')
        decoder_target_data = np.zeros(
                                   (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
                                   dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens
