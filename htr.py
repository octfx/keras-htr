import tensorflow as tf
import argparse
from keras_htr.char_table import CharTable
from keras_htr.models.htr_model import HTRModel
from keras_htr import codes_to_string
from keras_htr.preprocessor import Augmentor
import cv2

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
    adapter = model.get_adapter()


    img = Augmentor.preprocess(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (128, 32), augment=False)
    img = adapter.adapt_x(img)

    labels = model.predict(img)[0]
    res = codes_to_string(labels, char_table)

    print('Recognized text: "{}"'.format(res))
