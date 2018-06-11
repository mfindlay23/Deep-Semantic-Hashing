from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras import backend as K
import sys
import utils
import word2vec
import gensim
import word2vec

def get_session(gpu_num="2", gpu_fraction=0.2):
    import tensorflow as tf
    import os

    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class seq2seq(object):
    def __init__(self, latent_dim, num_encoder_tokens, num_decoder_tokens):
        self._training_model = None
        self._hashing_model = None
        self.hashing_inputs = None
        self.hashing_states = None
        self.latent_dim = latent_dim
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens

    @property
    def training_model(self):
        return self._training_model

    @training_model.setter
    def training_model(self, val):
        if self._training_model == None:
            self._training_model = val
        else:
            print("Attempted to train model twice, exiting...")
            sys.exit(0)

    @property
    def hashing_model(self):
        return self._hashing_model

    @hashing_model.setter
    def hashing_model(self, val):
        if self._hashing_model == None:
            self._hashing_model = val
        else:
            print("Hashing model created twice, exiting...")
            sys.exit(0)

    def _build_training(self):
        #Encoder portion
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        #decoder portion
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.hashing_inputs = encoder_inputs
        self.hashing_states = encoder_states
        self.training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def _build_hashing_function(self):
        self.hashing_model = Model(self.hashing_inputs, self.hashing_states)

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data,
                optimizer="rmsprop", loss='categorical_crossentropy', batch_size=100, epochs=10, validation_split=0.2):
        self._build_training()
        self.training_model.compile(optimizer=optimizer, loss=loss)
        self.training_model.fit([encoder_input_data, decoder_input_data],
            decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
        self.training_model.save("s2s_training_weights.h5")

    def hash(self, input_data):
        if self.hashing_model == None:
            self._build_hashing_function()
        states = self.hashing_model.predict(input_data)
        return states



if __name__ == "__main__":

    K.set_session(get_session())
    data_path = '20news-bydate-train'
    utils.DataUtils.get_20news_dataset(data_path)

    batch_size = 64  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000  # Number of samples to train on.
    # Path to the data txt file on disk.
    data_path = 'fra-eng/fra.txt'

    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    for doc in utils.DataUtils.X_train[:1000]:
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

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

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

    seq_model = seq2seq(latent_dim, num_encoder_tokens, num_decoder_tokens)
    seq_model.train(encoder_input_data, decoder_input_data, decoder_target_data)

    for seq_index in range(100):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        states_value = seq_model.hash(input_seq)
        print("Hash value: \n", states_value)
