import random

import cv2
import numpy as np


class Augmentor:

    @staticmethod
    def preprocess(img, image_size, augment=False):
        """put img into target img of size imgSize, transpose for TF and normalize gray-values"""

        def rand_odd():
            return random.randint(1, 3) * 2 + 1

        # there are damaged files in IAM dataset - just use black image instead
        if img is None:
            img = np.zeros(image_size[::-1])

        # data augmentation
        img = img.astype(np.float)
        if augment:
            # photometric data augmentation
            if random.random() < 0.25:
                img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)
            if random.random() < 0.25:
                img = cv2.dilate(img, np.ones((3, 3)))
            if random.random() < 0.25:
                img = cv2.erode(img, np.ones((3, 3)))
            if random.random() < 0.5:
                img = img * (0.25 + random.random() * 0.75)
            if random.random() < 0.25:
                img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 50), 0, 255)
            if random.random() < 0.1:
                img = 255 - img

            # geometric data augmentation
            wt, ht = image_size
            h, w = img.shape
            f = min(wt / w, ht / h)
            fx = f * np.random.uniform(0.75, 1.25)
            fy = f * np.random.uniform(0.75, 1.25)

            # random position around center
            txc = (wt - w * fx) / 2
            tyc = (ht - h * fy) / 2
            freedom_x = max((wt - fx * w) / 2, 0) + wt / 10
            freedom_y = max((ht - fy * h) / 2, 0) + ht / 10
            tx = txc + np.random.uniform(-freedom_x, freedom_x)
            ty = tyc + np.random.uniform(-freedom_y, freedom_y)


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
        img = cv2.warpAffine(img, M, dsize=image_size, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

        # convert to range [-1, 1]
        img = img / 255 - 0.5

        return img
