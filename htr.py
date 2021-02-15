import tensorflow as tf
import argparse
from project.htr import CharTable
from project.htr.models import HTRModel
from project.htr.preprocessor import Augmentor
import numpy as np
import cv2
gpus = tf.config.list_physical_devices('GPU')
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

def adapt_image(image):
    a = tf.keras.preprocessing.image.img_to_array(image)
    x = a / 255.0

    X = np.array(x).reshape(1, *x.shape)

    input_lengths = np.array(16, dtype=np.int32).reshape(len(X), 1)

    return X, input_lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('char_table', type=str)
    parser.add_argument('image', type=str)
    parser.add_argument('--raw', type=bool, default=False)

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    char_table_path = args.char_table
    raw = args.raw

    char_table = CharTable(char_table_path)

    model = HTRModel.load(model_path)

    img = Augmentor.preprocess(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (128, 32), augment=False)
    img = adapt_image(img)

    labels = model.predict(img)[0]

    print(labels)

    res = ''.join([char_table.get_character(label) for label in labels])

    print('Recognized text: "{}"'.format(res))
