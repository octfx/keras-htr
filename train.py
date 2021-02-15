import os
import argparse
from project.htr.generators import WordGenerator
from project.htr.models.htr_model import HTRModel
from tensorflow.keras.callbacks import Callback
from project.htr.char_table import CharTable
import tensorflow as tf
from project.htr.loader import DataLoaderIAM


def fix_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


class SaveModelCallback(Callback):
    def __init__(self, model, save_path):
        super().__init__()
        self._model = model
        self._save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self._model.save(self._save_path)


def fit_model(args, dataloader):
    data_source = args.data_source
    model_save_path = args.model_path
    batch_size = 1
    lr = args.learning_rate
    epochs = args.epochs

    char_table_path = os.path.join(data_source, 'characters.txt')

    char_table = CharTable(char_table_path)

    model = HTRModel(label_count=char_table.size)

    train_generator = WordGenerator(dataloader.get_train_samples(), char_table, batch_size, augment=args.augment)

    val_generator = WordGenerator(dataloader.get_validation_samples(), char_table, batch_size, augment=False)

    checkpoint = SaveModelCallback(model, model_save_path)

    model.fit(train_generator, val_generator, epochs=epochs, learning_rate=lr, callbacks=[checkpoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='temp_ds')
    parser.add_argument('--model_path', type=str, default='conv_lstm_model')
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--augment', type=bool, default=False)

    args = parser.parse_args()

    dataloader = DataLoaderIAM(args.data_source)
    fix_gpu()
    fit_model(args, dataloader)
