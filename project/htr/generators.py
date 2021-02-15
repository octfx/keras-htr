import random
import cv2
from .preprocessor.image_augmentor import Augmentor
import numpy as np


def compute_input_lengths(image_arrays):
    batch_size = len(image_arrays)
    return np.array(16, dtype=np.int32).reshape(batch_size, 1)


def adapt_batch(batch):
    image_arrays, labellings = batch

    current_batch_size = len(labellings)

    X = np.array(image_arrays).reshape(current_batch_size, *image_arrays[0].shape)

    labels = np.array(labellings, dtype=np.int32).reshape(current_batch_size, -1)

    input_lengths = compute_input_lengths(image_arrays)

    label_lengths = np.array([len(labelling) for labelling in labellings],
                             dtype=np.int32).reshape(current_batch_size, 1)

    return [X, labels, input_lengths, label_lengths], labels


class WordGenerator:
    """
    The dataset generator used by the model
    Outputs single words with corresponding labels
    If augment is set to true, images get augmented via the Augmentor class
    """

    def __init__(self, samples, char_table, batch_size=1, augment=True):
        self._samples = samples
        self._char_table = char_table
        self._batch_size = batch_size
        self._augment = augment

        self._indices = list(range(len(samples)))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def size(self):
        return len(self._samples)

    def __iter__(self):
        while True:
            for batch in self.get_batches():
                yield adapt_batch(batch)

    def get_batches(self):
        random.shuffle(self._indices)
        image_arrays = []
        labellings = []
        for line_index in self._indices:
            image_array, labels = self.get_dataset(line_index)
            image_arrays.append(image_array)
            labellings.append(labels)

            if len(labellings) >= self._batch_size:
                batch = image_arrays, labellings
                image_arrays = []
                labellings = []
                yield batch

        if len(labellings) >= 1:
            yield image_arrays, labellings

    def get_dataset(self, line_index):
        sample = self._samples[line_index]
        x = Augmentor.preprocess(cv2.imread(sample['file_path'], cv2.IMREAD_GRAYSCALE), (128, 32), self._augment)
        y = [self._char_table.get_label(ch) for ch in sample['text']]

        return x, y
