from graph_adt import GraphADT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time
from skimage.color import rgb2lab, lab2rgb, rgb2hsv
import os


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    codebook = [i for i in range(len(codebook))]
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    print('recreated image')
    return image


def recreate_2_image(labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = labels[label_idx]
            label_idx += 1
    return image


def normalize(img):
    sample = shuffle(np.reshape(img, (img.shape[0] * img.shape[1], 3)))[:min(500, img.shape[0] * img.shape[1])]
    img[:, :, 0] = (img[:, :, 0] - np.mean(sample[:, 0]))
    img[:, :, 1] = (img[:, :, 1] - np.mean(sample[:, 1]))
    img[:, :, 2] = (img[:, :, 2] - np.mean(sample[:, 2]))
    return img


def quantize_a_piece(window):
    width, height, depth = window.shape
    normalize(window)
    image_array = np.reshape(window, (width * height, depth))
    image_array[np.isnan(image_array)] = 0
    image_array_sample = shuffle(image_array)[:min(max(400, width * height), 1000)]
    kmeans = KMeans(n_clusters=2).fit(image_array_sample)
    labels = kmeans.predict(image_array)

    if np.sum(labels) > len(labels) // 2:
        if len(labels) - np.sum(labels) < len(labels) // 6:
            labels = np.where(labels == 0, 3, labels)
            labels = np.where(labels == 1, 0, labels)
            labels = np.where(labels == 3, 1, labels)
    return recreate_2_image(labels, width, height)


def quantize(img, h=4, w=4):
    img = np.array(img, dtype=np.float64)

    h, w = img.shape[0] // h, img.shape[1] // w
    newimg = np.zeros(shape=img.shape)

    for i in range(0, img.shape[0], h):
        for j in range(0, img.shape[1], w):
            window = img[i:i + h, j:j + w]
            newimg[i:i + h, j:j + w] = normalize(window)

    return quantize_a_piece(newimg)


def read_image():
    i_name = input('Enter name of an image file(must be in current directory): ')
    while i_name not in os.listdir():
        i_name = input('Enter name of an image file(must be in current directory) <"end" to break>: ').strip()
        if i_name != "end":
            return
    # to make double color picture \(x_0)/
    # but it works 
    img = plt.imread(i_name)
    img = plt.imread(i_name)
    return img


def main():
    img = read_image()
    if img is None:
        print('[INFO]: Wrong data')
        return
    # img = plt.imread('graph_225x225.png')
    points = quantize(img)
    graph = GraphADT(points)
    # auto converting to x and y and Array2D
    # so, need only to plot some images
    graph.show()
    graph.simple_reveal()
    graph.reveal()
    graph.to_csv(f_name='result.csv')


if __name__ == '__main__':
    main()
