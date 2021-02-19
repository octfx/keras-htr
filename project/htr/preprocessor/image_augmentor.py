import random

import cv2
import numpy as np


class Augmentor:

    @staticmethod
    def preprocess(img, image_size, augment=False, binarize=False):
        """
        Augment an input image
        - 33%: Rotate between -25° and 25°
        - 33%: Affine translate on the x and y axis between -20% and 20%
        - 33%: Rotation and Translation

        Images get binarized by using otsus method

        Output image has a size of (128, 32)
        Grayscale values are put into the range [0, 1]
        :param binarize:
        :param img:
        :param image_size:
        :param augment:
        :return:
        """

        rotation_range = (-25, 25)
        scaling_range = (0.5, 1.5)

        original = img.copy()

        if img is None:
            img = np.zeros(image_size[::-1])

        if binarize:
            # Binarization flag is usually set when preprocessing an image for inference
            # Images in lmdb are already thresholded
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = img.astype(np.float)

        if augment:
            # Augmentation is usually done when training the network
            scale = 1
            scale_done = False
            rotation = 0
            rotation_done = False

            if random.random() < 0.33:
                rotation = random.randint(rotation_range[0], rotation_range[1])
                rotation_done = True
            if random.random() < 0.33:
                scale = random.uniform(scaling_range[0], scaling_range[1])
                scale_done = True
            if random.random() < 0.33:
                if not rotation_done:
                    rotation = random.randint(rotation_range[0], rotation_range[1])
                if not scale_done:
                    scale = random.uniform(scaling_range[0], scaling_range[1])

            # The matrix used for rotation and scaling
            M = cv2.getRotationMatrix2D(
                (img.shape[0] // 2, img.shape[1] // 2),
                angle=rotation,
                scale=scale
            )

            # The image is rotated and scaled
            img = cv2.warpAffine(
                img,
                M,
                (img.shape[1], img.shape[0]),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

            target_width, target_height = image_size
            original_height, original_width = img.shape
            f = min(target_width / original_width, target_height / original_height)
            fx = f * np.random.uniform(0.75, 1.25)
            fy = f * np.random.uniform(0.75, 1.25)

            # random position around center
            txc = (target_width - original_width * fx) / 2
            tyc = (target_height - original_height * fy) / 2
            tx = txc + np.random.uniform(-0.2, 0.2)
            ty = tyc + np.random.uniform(-0.2, 0.2)

        else:
            # center image
            target_width, target_height = image_size
            original_height, original_width = img.shape
            f = min(target_width / original_width, target_height / original_height)
            tx = (target_width - original_width * f) / 2
            ty = (target_height - original_height * f) / 2

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
            # If the augemtation yields an empty image try again
            return Augmentor.preprocess(original, image_size, augment)

        # if random.random() < 0.001:
        #    cv2.imwrite('check/{}.png'.format(random.randint(100, 10000)), img)

        # convert to range [0, 1]
        img = img / 255.0

        assert np.min(img) >= 0 and np.max(img) <= 1, "Image not in range [0, 1]. How could this happen"

        return img
