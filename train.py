import argparse
import os
import pickle

import cv2
from path import Path
from tensorflow.keras.callbacks import Callback

import lmdb
from project.htr.char_table import CharTable
from project.htr.loader import DataLoaderIAM
from project.htr.models import limit_gpu_memory
from project.htr.models.htr_model import HTRModel
from project.htr.word_generator import WordGenerator


#
# python train.py --model_path=model --learning_rate=0.001 --augment=true --epochs=100 --batch_size=150
#


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
