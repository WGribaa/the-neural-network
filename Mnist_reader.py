import struct
import numpy as np


def read_image_file(imagefile, flatten=False):
    """
    Reads an idx file and returns the pixel matrix.
    :param imagefile: idx file to read as a filepath.
    :param flatten: Flatten each image if True.
    :return: A 3d ndarray :
        1- Index of the image in the dataset.
        2- Index of the row.
        3- Index of the column.
    """
    with open(imagefile, "br") as file:
        magic_number, n_images = read_bytes(file.read(8))
        n_rows, n_cols = read_bytes(file.read(8))
        data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
        print("%i images of size %i rows * %i columns." % (n_images, n_rows, n_cols))
        data = data.reshape((n_images, n_rows, n_cols))

    if flatten:
        return flatten_matrix(data, n_cols*n_rows)
    else:
        return data


def read_label_file(labelfile):
    """
    Reads an idx file and returns the target matrix.
    :param labelfile: idx file to read as a filepath.
    :return: A 3d ndarray :
        1- Index of the image in the dataset.
        2- Index of the row.
        3- Index of the column.
    """
    with open(labelfile, "br") as file:
        magic_number, n_images = read_bytes(file.read(8))
        data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape(n_images)
        return data


def flatten_matrix(mat, length):
    return [elem.reshape(length) for elem in mat]


def read_bytes(chunk):
    """
    Reads a chunk of 2 bytes and returns each as a tuple.
    :param chunk: 2 bytes tor read.
    :return: A tuple of bytes
    """
    byte1, byte2 = struct.unpack(">II", chunk)
    return int(byte1), int(byte2)