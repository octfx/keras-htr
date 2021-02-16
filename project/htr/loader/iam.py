import os
import random


class DataLoaderIAM:
    """
    loads data which corresponds to IAM format, see:
    http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    _train_samples = None
    _validation_samples = None

    @staticmethod
    def truncate_label(text, max_text_length):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_length:
                return text[:i]
        return text

    def __init__(self, data_dir):
        """
        Loads the IAM Dataset to lmdb
        Splits the dataset into training and validation
        :param data_dir:
        """

        max_text_len = 16  # Max length for a word (label)

        assert os.path.isdir(os.path.join(data_dir, 'img'))

        self.samples = []

        self.data_dir = data_dir

        words = open(os.path.join(data_dir, 'words.txt'))
        character_vocabulary = set()
        for line in words:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            file_name_split = line_split[0].split('-')
            file_name = os.path.join(
                data_dir,
                'img',
                file_name_split[0],
                '{}-{}'.format(file_name_split[0], file_name_split[1]),
                line_split[0] + '.png'
            )

            # Bad files
            if line_split[0] in ['a01-117-05-02', 'r06-022-03-05']:
                continue

            # Text in words.txt start at index 9
            image_text = DataLoaderIAM.truncate_label(' '.join(line_split[8:]), max_text_len)
            character_vocabulary = character_vocabulary.union(set(list(image_text)))

            # put sample into list
            # self.samples.append(DatasetSample(image_text, file_name))
            self.samples.append(dict({
                'text': image_text,
                'file_path': file_name
            }))

        self.split_dataset()

        self.characters = sorted(list(character_vocabulary))

        with open(os.path.join(data_dir, 'dict.txt'), 'w') as f:
            f.write(self.samples.__str__())

    def write_characters_to_disk(self):
        """
        Writes the set of found characters to disk
        :return:
        """
        with open(os.path.join(self.data_dir, 'characters.txt'), 'w') as f:
            out = ''
            for char in self.characters:
                out += char
            f.write(out)

    def split_dataset(self):
        """Split the dataset into training and validation"""
        split_index = int(0.95 * len(self.samples))
        self._train_samples = self.samples[:split_index]
        self._validation_samples = self.samples[split_index:]

    def get_train_samples(self):
        random.shuffle(self._train_samples)

        return self._train_samples

    def get_validation_samples(self):
        return self._validation_samples
