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

    encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens = utils.DataUtils.prep_train_data(0, 1000)
    seq_model = seq2seq(latent_dim, num_encoder_tokens, num_decoder_tokens)
    seq_model.train(encoder_input_data, decoder_input_data, decoder_target_data)

    #enc_test, _, _, _, _= utils.DataUtils.prep_train_data(1000, 1100)
    with open("hash_values.txt", "w") as f:
        for seq_index in range(100):
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            states_value = np.array(seq_model.hash(input_seq))
            hash_code = utils.hash(states_value)
            print("Hash value: \n", hash_code)
            f.write(''.join(str(hash_code)))
            f.write(',')
            f.write(str(utils.DataUtils.Y_train[seq_index]))
            f.write('\n')
