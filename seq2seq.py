from keras.models import Model
from keras.layers import Input, LSTM, Dense

class seq2seq(object):
    def __init__(self):
        pass

    def _build():
        #Define input seq and process
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        #Discard output state, unecessary
        encoder_states = [state_h, state_c]

        ######Decoder######
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(latent_dim, return_sequence=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        decoder_dense = Dense(num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model, encoder_inputs, decoder_inputs, decoder_outputs

    def train(data, optimizer="rmsprop", loss='categorical_crossentropy', batch_size='100', epochs='100', validation_split=0.2):
        model, encoder_inputs, decoder_inputs, decoder_outputs  = _build()
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

