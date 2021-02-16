import os
import argparse
import cv2
import lmdb
import pickle
from project.htr.word_generator import WordGenerator
from project.htr.models.htr_model import HTRModel
from tensorflow.keras.callbacks import Callback
from project.htr.char_table import CharTable
import tensorflow as tf
from project.htr.loader import DataLoaderIAM
from path import Path

#
# python train.py --model_path=model --learning_rate=0.001 --augment=true --epochs=80 --batch_size=10
#


def limit_gpu_memory(max_vram_mb=4096):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_vram_mb)])
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


def create_db(iam_source):
    if os.path.exists('lmdb'):
        return

    print("Loading images into LMDB...")

    data_dir = os.path.join(iam_source, 'img')

    # 2GB is enough for IAM dataset
    assert os.path.exists(data_dir), "IAM directory does not exist"

    env = lmdb.open('lmdb', map_size=1024 * 1024 * 1024 * 2)

    path = Path(data_dir)

    # go over all png files
    fn_imgs = list(path.walkfiles('*.png'))

    # and put the imgs into lmdb as pickled grayscale imgs
    with env.begin(write=True) as txn:
        for i, fn_img in enumerate(fn_imgs):
            if i % 1000 == 0:
                print("Loaded {} of {} images.".format(i, len(fn_imgs)))
            img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            basename = fn_img.basename()
            txn.put(basename.encode("ascii"), pickle.dumps(img))

    env.close()


def fit_model(args, dataloader):
    data_source = args.data_source
    model_save_path = args.model_path
    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs

    char_table_path = os.path.join(data_source, 'characters.txt')

    char_table = CharTable(char_table_path)

    model = HTRModel(label_count=char_table.size, model_path=model_save_path)

    train_generator = WordGenerator(dataloader.get_train_samples(), char_table, batch_size, augment=args.augment)

    val_generator = WordGenerator(dataloader.get_validation_samples(), char_table, batch_size, augment=False)

    checkpoint = SaveModelCallback(model, model_save_path)

    model.fit(
        train_generator,
        val_generator,
        epochs=epochs,
        learning_rate=lr,
        callbacks=[checkpoint],
        batch_size=batch_size
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='temp_ds')
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--augment', type=bool, default=False)

    max_vram = 4096

    args = parser.parse_args()

    create_db(args.data_source)
    dataloader = DataLoaderIAM(args.data_source)
    limit_gpu_memory(max_vram)
    fit_model(args, dataloader)
