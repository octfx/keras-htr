import os
import pickle
import random

import lmdb
import numpy as np
from path import Path

from .preprocessor.image_augmentor import Augmentor


def compute_input_lengths(image_arrays):
    lstm_input_shapes = [a.shape[1] // 8 for a in image_arrays]

    return np.array(
        lstm_input_shapes,
        dtype=np.int32
    ).reshape(
        len(image_arrays),
        1
    )


def pad_labellings(labels):
    """
    Zero pads all input labels to the longest one provided
    :param labels:
    :return:
    """
    target_length = max([len(labels) for labels in labels])
    padded = []

    for label in labels:
        padding_size = target_length - len(label)

        padded_label = label + [0] * padding_size

        assert len(padded_label) > 0

        padded.append(padded_label)

    return padded


def adapt_batch(batch):
    """
    Returns batchdata in the form of
    :param batch:
    :return:
    """
    image_arrays, labellings = batch

    current_batch_size = len(labellings)

    images = np.array(image_arrays).reshape(current_batch_size, *image_arrays[0].shape)

    padded_labellings = pad_labellings(labellings)

    labels = np.array(padded_labellings, dtype=np.int32).reshape(current_batch_size, -1)

    input_lengths = compute_input_lengths(image_arrays)

    label_lengths = np.array([len(labelling) for labelling in labellings],
                             dtype=np.int32).reshape(current_batch_size, 1)

    return [images, labels, input_lengths, label_lengths], labels


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

        assert os.path.isdir(os.path.join('lmdb'))

        self.env = lmdb.open('lmdb', readonly=True)

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
        """
        Retrieves data for usage in batches
        Images are loaded from lmdb
        :param line_index:
        :return:
        """
        sample = self._samples[line_index]

        with self.env.begin() as txn:
            basename = Path(sample['file_path']).basename()
            data = txn.get(basename.encode("ascii"))
            img = pickle.loads(data)

        x = Augmentor.preprocess(img, (128, 32), self._augment)
        y = [self._char_table.get_label(ch) for ch in sample['text']]

        return x, y
