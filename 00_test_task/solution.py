import os
import sys
import math
import time
import operator

import numpy as np

from PIL import Image
from functools import reduce
from argparse import ArgumentParser

import logging

_format = '%(message)s'
logging.basicConfig(level=logging.DEBUG, format=_format)
logger = logging.getLogger(__name__)


def path_processing(args):
    path = args.path[0]
    if not os.path.exists(path):
        logger.debug("error: No such \"{}\" directory found!".format(path))
        sys.exit()
    else:
        create_image_pairs(path)


def create_image_pairs(path):
    _path = path
    image_pairs = []
    for (root, dirs, files) in os.walk(_path, topdown=False):
        for name in files:
            for (root_, dirs_, files_) in os.walk(_path, topdown=False):
                for name_ in files_:
                    p_1, p_2 = os.path.join(root, name), os.path.join(root_, name_)
                    if p_1 == p_2 or [p_2, p_1] in image_pairs:
                        continue
                    image_pairs.append([p_1, p_2])
    begin_compare(image_pairs)


def image_duplicate(filepath1, filepath2):
    image1 = get_thumbnail(Image.open(filepath1))
    image2 = get_thumbnail(Image.open(filepath2))

    if image1.size != image2.size or image1.getbands() != image2.getbands():
        return -1
    s = 0
    for band_index, band in enumerate(image1.getbands()):
        m1 = np.array([p[band_index] for p in image1.getdata()]).reshape(*image1.size)
        m2 = np.array([p[band_index] for p in image2.getdata()]).reshape(*image2.size)
        s += np.sum(np.abs(m1 - m2))
    return s


def image_histogram(filepath1, filepath2):
    ac0, ac1, ac2 = 55, 0.95, 1200
    image1 = Image.open(filepath1)
    image2 = Image.open(filepath2)

    h1 = get_thumbnail(image1).histogram()
    h2 = get_thumbnail(image2).histogram()

    rms = math.sqrt(reduce(operator.add, list(map(lambda a, b:
                                                  (a - b) ** 2, h1, h2))) / len(h1))
    # reserved for estimating values ​​of explicit modifications
    # and (image1.size != image2.size or image1.getbands() != image2.getbands()):
    if rms <= ac0:
        # reserved for via numpy
        # res = numpy_similarity(image1, image2)
        # return (0, res) if (res > ac1) else (1, res)
        res = greyscale_hash_code_similarity(image1, image2)
        return (0, res) if (res < ac2) else (1, res)
    return -1, rms


def greyscale_hash_code_similarity(image1, image2):
    code1 = image_pixel_hash_code(get_thumbnail(image1, greyscale=True))
    code2 = image_pixel_hash_code(get_thumbnail(image2, greyscale=True))
    res = hamming_distance(code1, code2)
    return res


def image_pixel_hash_code(image):
    pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join(map(lambda pixel: '1' if pixel < avg else '0', pixels))
    hexadecimal = int(bits, 2).__format__('016x').upper()
    return hexadecimal


def hamming_distance(s1, s2):
    len1, len2 = len(s1), len(s2)
    if len1 != len2:
        if len1 > len2:
            s1 = s1[:-(len1 - len2)]
        else:
            s2 = s2[:-(len2 - len1)]
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


def numpy_similarity(image1, image2):
    _image1 = get_thumbnail(image1, size=(128, 128), stretch_to_fit=True)
    _image2 = get_thumbnail(image2, size=(128, 128), stretch_to_fit=True)

    images = [_image1, _image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        norms.append(np.linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = np.dot(a / a_norm, b / b_norm)
    return res


def get_thumbnail(image, size=(160, 160), stretch_to_fit=False, greyscale=False):
    if not stretch_to_fit:
        image.thumbnail(size, Image.LANCZOS)
    else:
        image = image.resize(size)
    if greyscale:
        image = image.convert("L")
    return image


def begin_compare(image_pairs):
    for pair in image_pairs:

        if not pair:
            continue

        if image_duplicate(*pair) == 0:
            logger.debug("{} {}".format(os.path.basename(pair[0]), os.path.basename(pair[1])))
            continue

        # t1 = time.time()
        sim = image_histogram(*pair)
        # _duration = "%0.1f" % ((time.time() - t1) * 1000)
        if sim[0] != -1:
            logger.debug("{} {}".format(os.path.basename(pair[0]), os.path.basename(pair[1])))

        # reserved for ~ handle types of similarity/modification
        # but other requirements in the:
        # "Example of solution interface"
        # if sim[0] == 0:
        #     logger.debug(" ~ image_modification => {} took {} ms".format(sim[1], _duration))
        # if sim[0] == 1:
        #     logger.debug(" == image_similar => {} took {} ms".format(sim[1], _duration))


def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, nargs=1, metavar="PATH",
                        default=None, help="folder with images")
    args = parser.parse_args()
    path_processing(args) if args.path \
        else logger.debug("error: the following arguments are required: --path")


if __name__ == '__main__':
    main()
