import random

import cv2
import numpy as np


class Augmentor:

    @staticmethod
    def preprocess(img, image_size, augment=False):
        """
        Augment an input image
        - 33%: Rotate between -25° and 25°
        - 33%: Affine translate on the x and y axis between -20% and 20%
        - 33%: Rotation and Translation

        Images get binarized by using otsus method

        Output image has a size of (128, 32)
        Grayscale values are put into the range [0, 1]
        :param img:
        :param image_size:
        :param augment:
        :return:
        """

        rotation_range = (-25, 25)
        scaling_range = (0.5, 1.5)

        original = img.copy()

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # there are damaged files in IAM dataset - just use black image instead
        if img is None:
            img = np.zeros(image_size[::-1])

        # data augmentation
        img = img.astype(np.float)

        if augment:
            scale = 1
            rotation = 0

            if random.random() < 0.33:
                rotation = random.randint(rotation_range[0], rotation_range[1])
            if random.random() < 0.33:
                scale = random.uniform(scaling_range[0], scaling_range[1])
            if random.random() < 0.33:
                rotation = random.randint(rotation_range[0], rotation_range[1])
                scale = random.uniform(scaling_range[0], scaling_range[1])

            M = cv2.getRotationMatrix2D(
                (img.shape[0] // 2, img.shape[1] // 2),
                angle=rotation,
                scale=scale
            )

            img = cv2.warpAffine(
                img,
                M,
                (img.shape[1], img.shape[0]),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            # geometric data augmentation
            wt, ht = image_size
            h, w = img.shape
            f = min(wt / w, ht / h)
            fx = f * np.random.uniform(0.75, 1.25)
            fy = f * np.random.uniform(0.75, 1.25)

            # random position around center
            txc = (wt - w * fx) / 2
            tyc = (ht - h * fy) / 2
            tx = txc + np.random.uniform(-0.2, 0.2)
            ty = tyc + np.random.uniform(-0.2, 0.2)

        # no data augmentation
        else:
            # center image
            wt, ht = image_size
            h, w = img.shape
            f = min(wt / w, ht / h)
            tx = (wt - w * f) / 2
            ty = (ht - h * f) / 2

        # map image into target image
        M = np.float32([[f, 0, tx], [0, f, ty]])

        target = np.ones(image_size[::-1]) * 255 / 2
        img = cv2.warpAffine(
            img,
            M,
            dsize=image_size,
            dst=target,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        if np.min(img) == 255:
            return Augmentor.preprocess(original, image_size, augment)

        # convert to range [0, 1]
        img = img / 255.0

        assert np.min(img) >= 0 and np.max(img) <= 1

        return img
