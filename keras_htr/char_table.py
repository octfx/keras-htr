import os
import io


def load_characters(path):
    char_to_label = {}
    label_to_char = {}

    with io.open(path, encoding='utf8') as f:
        index = 0
        for line in f.readlines():
            ch = line.rstrip('\n')
            for character in ch:
                char_to_label[character] = index
                label_to_char[index] = character
                index += 1

    return char_to_label, label_to_char


class CharTable:
    def __init__(self, characters_list_path):
        assert os.path.exists(characters_list_path)

        self._char_to_label, self._label_to_char = load_characters(characters_list_path)

        self._max_label = max(self._label_to_char.keys())

    @property
    def size(self):
        return len(self._char_to_label) + 2

    @property
    def sos(self):
        return self._max_label + 1

    @property
    def eos(self):
        return self.sos + 1

    def get_label(self, ch):
        return self._char_to_label[ch]

    def get_character(self, class_label):
        if class_label == self.sos:
            return ''

        if class_label == self.eos:
            return '\n'

        return self._label_to_char[class_label]
