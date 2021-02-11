import json
import math
import os

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.backend import ctc_batch_cost as CtcBatchCost
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow import nn


class HTRModel:
    lstm_units = 256

    def __init__(self, label_count, height=32):
        self._label_count = label_count
        self._height = height  # 32

        self._setup_rnn_layers()

    @staticmethod
    def create(model_path):
        model = HTRModel.load(model_path)

        return model

    @staticmethod
    def _create_conv_net():
        """
        Base
        Convolutional NN layers
        Feature Extraction
        16 -> 256 Features
        :return:
        """
        num_layers = 5
        kernel_size = (3, 3)

        model = Sequential()

        # Dropout (layers 2+)
        # Conv2D
        # LeakyReLU
        # MaxPool (layers 1-3)
        # BatchNormalization
        for idx in range(num_layers):
            num_filters = 2 ** (idx + 4)  # 16 -> 256
            print(num_filters)

            if idx > 2:
                model.add(Dropout(rate=0.15))

            if idx == 0:
                model.add(
                    Conv2D(
                        input_shape=(None, None, 1),
                        filters=num_filters,
                        kernel_size=kernel_size,
                        padding='same',
                        activation=None
                    ))
            else:
                model.add(Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same', activation=None))

            model.add(LeakyReLU())

            if idx < 3:
                model.add(MaxPool2D())

            model.add(BatchNormalization())

        def concat(X):
            t = Concatenate(axis=1)(tf.unstack(X, axis=3))
            t = tf.transpose(t, [0, 2, 1])

            return t

        model.add(Lambda(concat))

        return model

    def _setup_rnn_layers(self):
        """
        Recurrent NN Layers
        256 Features per Time Step
        Adds Dropout layers to make the model more robust
        :return:
        """
        num_layers = 5

        # Input grayscale images with height 32px and width 128px
        self.input_layer = Input(shape=(self._height, None, 1))

        model = self.input_layer
        # Keras functional API
        model = self._create_conv_net()(model)

        model = Dropout(rate=0.5)(model)
        model = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(model)

        for idx in range(num_layers):
            model = Dropout(rate=0.5)(model)
            model = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(model)

        model = Dropout(rate=0.5)(model)

        self.output_layer = TimeDistributed(Dense(units=self._label_count + 1, activation='softmax'))(model)
        self.weights = Model(self.input_layer, self.output_layer)

    def _make_infer(self):
        """
        THe model used for inference
        :return:
        """
        return Model(self.input_layer, self.output_layer)

    def _make_train(self):
        """
        The model used for training
        Runs Connectionist Temporal Classification Loss algorithm on each batch element.
        :return:
        """

        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            return CtcBatchCost(labels, y_pred, input_length, label_length)

        labels = Input(name='the_labels', shape=[None], dtype=tf.float32)
        input_length = Input(name='input_length', shape=[1], dtype=tf.int64)
        label_length = Input(name='label_length', shape=[1], dtype=tf.int64)

        return Model(
            inputs=[self.input_layer, labels, input_length, label_length],
            # Lambda layer to use keras backend function
            outputs=Lambda(
                ctc_lambda_func,
                output_shape=(1,),
                name='CTC_Loss'
            )([self.output_layer, labels, input_length, label_length])
        )

    def fit(self, train_data, validation_data, epochs, learning_rate, callbacks):
        steps_per_epoch = math.ceil(train_data.size / train_data.batch_size)
        val_steps = math.ceil(validation_data.size / validation_data.batch_size)

        learning_rate = learning_rate or 0.001
        epochs = epochs or 100
        callbacks = callbacks or []

        training_model = self._make_train()

        training_model.compile(
            optimizer=Adam(lr=learning_rate),
            loss={'CTC_Loss': lambda true, predicted: predicted}
        )

        training_model.fit(
            train_data.__iter__(),
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data.__iter__(),
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks
        )

    def get_preprocessor(self):
        from keras_htr.preprocessing import Cnn1drnnCtcPreprocessor
        preprocessor = Cnn1drnnCtcPreprocessor()
        # preprocessor.configure(**self._preprocessing_options)
        return preprocessor

    def get_adapter(self):
        from ..adapters.cnn_1drnn_ctc_adapter import CTCAdapter
        return CTCAdapter()

    def predict(self, inputs):
        X, input_lengths = inputs
        #X = inputs

        prediction = self._make_infer().predict(X)

        labels = decode_greedy(prediction, input_lengths)

        return labels

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

        params_path = os.path.join(path, 'params.json')
        weights_path = os.path.join(path, 'weights.h5')

        with open(params_path, 'w') as f:
            f.write(json.dumps({
                'params': dict(
                    label_count=self._label_count
                )
            }))

        self.weights.save_weights(weights_path)

        inference_model = self._make_infer()

        inference_model_path = os.path.join(path, 'inference_model.h5')
        inference_model.save(inference_model_path)

    @classmethod
    def load(cls, path):
        params_path = os.path.join(path, 'params.json')
        weights_path = os.path.join(path, 'weights.h5')

        with open(params_path) as f:
            s = f.read()

        param_data = json.loads(s)

        instance = cls(label_count=param_data['params']['label_count'], height=32)  # TODO change height

        instance.weights.load_weights(weights_path)
        return instance


@tf.function
def decode_greedy(inputs, input_lengths):
    inputs = tf.transpose(inputs, [1, 0, 2])
    decoded, _ = nn.ctc_greedy_decoder(inputs, input_lengths.flatten())

    dense = tf.sparse.to_dense(decoded[0])

    return dense.numpy()


@tf.function
def beam_search_decode(inputs, input_lengths):
    inputs = tf.transpose(inputs, [1, 0, 2])
    decoded, log_probs = nn.ctc_beam_search_decoder(inputs, input_lengths.flatten(), beam_width=10)
    print(log_probs)
    dense = tf.sparse.to_dense(decoded[0])

    return dense.numpy()
