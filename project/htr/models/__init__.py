import os

import cv2
import numpy as np
import tensorflow as tf

from .htr_model import HTRModel
from ..char_table import CharTable
from ..preprocessor.image_augmentor import Augmentor


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


def predict(model_path, char_table, image, decode_mode='Greedy'):
    limit_gpu_memory(1024)

    assert os.path.exists(model_path), "Model path not found"
    assert os.path.exists(char_table), "Character table not found"

    char_table = CharTable(char_table)

    model = HTRModel.load(model_path)

    if type(image) == str:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = Augmentor.preprocess(image, (128, 32), augment=False, binarize=True)
    else:
        img = image

    X = np.array(img).reshape(1, *img.shape)

    input_lengths = np.array(16, dtype=np.int32).reshape(1, 1)

    labels = model.predict(
        [X, input_lengths],
        decode_mode=decode_mode
    )[0]

    return ''.join([char_table.get_character(label) for label in labels])
