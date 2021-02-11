import os
import logging
from keras_htr import get_meta_info
from keras_htr.generators import LinesGenerator, WordGenerator
from keras_htr.models.htr_model import HTRModel
from tensorflow.keras.callbacks import Callback
from keras_htr.char_table import CharTable
import tensorflow as tf
import json
from pathlib import Path
from keras_htr.loader import DataLoaderIAM

def fix_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
    logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


class MyModelCheckpoint(Callback):
    def __init__(self, model, save_path):
        super().__init__()
        self._model = model
        self._save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self._model.save(self._save_path)



def fit_model(args, dataloader):
    dataset_path = args.ds
    model_save_path = args.model_path
    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs
    augment = args.augment

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'validation')

    #meta_info = get_meta_info(path=train_path)

    #image_height = meta_info['average_height']
    #image_height = 32

    char_table_path = os.path.join(dataset_path, 'characters.txt')

    char_table = CharTable(char_table_path)

    model = HTRModel(label_count=char_table.size)

    adapter = model.get_adapter()

    train_generator = WordGenerator(dataloader.get_train_samples(), char_table, batch_size, augment=True)

    val_generator = WordGenerator(dataloader.get_validation_samples(), char_table, batch_size, augment=False)
    # train_generator = LinesGenerator(train_path, char_table, batch_size,
    #                                  augment=augment, batch_adapter=adapter)
    #
    # val_generator = LinesGenerator(val_path, char_table, batch_size,
    #                                batch_adapter=adapter)

    checkpoint = MyModelCheckpoint(model, model_save_path)


    fix_gpu()
    model.fit(train_generator, val_generator, epochs=epochs, learning_rate=lr, callbacks=[checkpoint])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default='temp_ds')
    parser.add_argument('--model_path', type=str, default='conv_lstm_model')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--augment', type=bool, default=False)

    args = parser.parse_args()

    dataloader = DataLoaderIAM('temp_ds')
    fit_model(args, dataloader)
