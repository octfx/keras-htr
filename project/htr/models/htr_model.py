import math
import os

import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import Model, Sequential
from tensorflow.keras.backend import ctc_batch_cost as CtcBatchCost
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam


class HTRModel:
    lstm_units = 256
    image_height = 32
    image_width = 128

    def __init__(self, label_count=80, model_path=None):
        self._label_count = label_count
        self._model_path = model_path

        self._setup_layers()
        print("Summary:")
        print(self.weights.summary())

    @staticmethod
    def create(model_path):
        """
        Create a model by loading weights from the specified path
        :param model_path:
        :return:
        """
        model = HTRModel.load(model_path)

        return model

    @staticmethod
    def _create_conv_net():
        """
        Base
        Convolutional NN layers
        Feature Extraction
        16 -> 80 Features
        :return:
        """
        num_layers = 5
        kernel_size = (3, 3)

        model = Sequential()

        # Dropout (layers 2+)
        # Conv2D
        # BatchNormalization
        # LeakyReLU
        # MaxPool (layers 1-3)
        for idx in range(num_layers):
            num_filters = 16 * (idx + 1)  # 16 -> 80
            # print(num_filters)

            if idx > 1:
                model.add(Dropout(rate=0.2))

            if idx == 0:
                model.add(
                    Conv2D(
                        input_shape=(None, None, 1),
                        filters=num_filters,
                        kernel_size=kernel_size,
                        strides=(1, 1),
                        padding='same',
                        activation=None
                    )
                )
            else:
                model.add(
                    Conv2D(
                        filters=num_filters,
                        kernel_size=kernel_size,
                        strides=(1, 1),
                        padding='same',
                        activation=None
                    )
                )

            model.add(BatchNormalization())
            model.add(LeakyReLU())

            if idx < 3:
                model.add(MaxPool2D())

        def columnwise_concat(layer):
            transposed = Concatenate(axis=1)(tf.unstack(layer, axis=3))
            transposed = tf.transpose(transposed, [0, 2, 1])

            return transposed

        model.add(Lambda(columnwise_concat))

        return model

    def _setup_layers(self):
        """
        Recurrent NN Layers
        256 Features per Time Step
        Adds Dropout layers to make the model more robust
        :return:
        """
        num_layers = 5

        # Input grayscale images with height 32px and width 128px
        self.input_layer = Input(shape=(self.image_height, self.image_width, 1))

        model = self.input_layer
        # Keras functional API
        model = self._create_conv_net()(model)

        model = Dropout(rate=0.5)(model)

        for idx in range(num_layers):
            model = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=0.5))(model)

        model = Dropout(rate=0.5)(model)

        self.output_layer = TimeDistributed(Dense(units=self._label_count, activation='softmax'))(model)

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

    def fit(self, train_data, validation_data, epochs, learning_rate, callbacks, batch_size):
        """
        Fit the model to the provided training data
        :param batch_size:
        :param train_data:
        :param validation_data:
        :param epochs:
        :param learning_rate:
        :param callbacks:
        :return:
        """
        steps_per_epoch = math.ceil(train_data.size / train_data.batch_size)
        val_steps = math.ceil(validation_data.size / validation_data.batch_size)

        learning_rate = learning_rate or 0.001
        epochs = epochs or 100
        callbacks = callbacks or []

        training_model = self._make_train()

        if self._model_path is not None and os.path.exists(os.path.join(self._model_path, 'weights.h5')):
            # If previously saved weights are available, further train on them
            print("Using previously saved weights.")
            training_model.load_weights(os.path.join(self._model_path, 'weights.h5'))

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
            callbacks=callbacks,
            batch_size=batch_size
        )

    def predict(self, inputs):
        image, input_lengths = inputs
        prediction = self._make_infer().predict(image)

        labels = greedy_decode(prediction, input_lengths)
        # labels = beam_search_decode(prediction, input_lengths)

        return labels

    def save(self, path):
        """
        Writes the computed weights to disk
        :param path:
        :return:
        """
        if not os.path.exists(path):
            os.mkdir(path)

        weights_path = os.path.join(path, 'weights.h5')

        self.weights.save_weights(weights_path)

    @classmethod
    def load(cls, path):
        weights_path = os.path.join(path, 'weights.h5')

        assert os.path.exists(weights_path)

        instance = cls()

        instance.weights.load_weights(weights_path)
        return instance


def beam_search_decode(inputs, input_lengths):
    with tf.compat.v1.Session() as sess:
        decoded, _ = nn.ctc_beam_search_decoder(
            inputs=tf.transpose(inputs, [1, 0, 2]),
            sequence_length=input_lengths.flatten(),
            beam_width=10,
            merge_repeated=True
        )

        dense = tf.sparse.to_dense(decoded[0])
        res = sess.run(dense)
        return res


def greedy_decode(inputs, input_lengths):
    with tf.compat.v1.Session() as sess:
        decoded, _ = nn.ctc_greedy_decoder(
            inputs=tf.transpose(inputs, [1, 0, 2]),
            sequence_length=input_lengths.flatten(),
            merge_repeated=True
        )

        dense = tf.sparse.to_dense(decoded[0])
        res = sess.run(dense)
        return res
