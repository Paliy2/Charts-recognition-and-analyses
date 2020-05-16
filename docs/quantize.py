import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 64

def cluster_colors(image_array_sample, n_colors):
    kmeans = KMeans(n_clusters=n_colors).fit(image_array_sample)
    return kmeans

def quantize_binary(window):
    width, height, depth = window.shape
    image_array = np.reshape(window, (width * height, depth))
    image_array[np.isnan(image_array)] = 0
    image_array_sample = shuffle(image_array)[:min(600, width * height)]

    kmeans = cluster_colors(image_array_sample, 2)
    labels = kmeans.predict(image_array)

    if np.sum(labels) > len(labels)//2:
            labels = np.where(labels==0, 3, labels)
            labels = np.where(labels==1, 0, labels)
            labels = np.where(labels==3, 1, labels)
    return labels.reshape((width, height))


def normalize(img):
    sample = shuffle(np.reshape(img, (img.shape[0]*img.shape[1], 3)))[:min(500,img.shape[0]*img.shape[1])]
    img[:, :, 0] = (img[:, :, 0] - np.mean(sample[:, 0]))
    img[:, :, 1] = (img[:, :, 1] - np.mean(sample[:, 1]))
    img[:, :, 2] = (img[:, :, 2] - np.mean(sample[:, 2]))
    return img


def normalize_by_region(image: np.array, h=4, w=4):
    img = np.copy(image)
    h, w = img.shape[0]//h, img.shape[1]//w
    for i in range(0, img.shape[0], h):
        for j in range(0, img.shape[1], w):
            img[i:i+h][j:j+w] = normalize(img[i:i+h][j:j+w])
    return img

def quantize(img, h=4, w=4):
    img = np.array(img, dtype=np.float64)

    h, w = img.shape[0]//h, img.shape[1]//w
    newimg = np.zeros(shape=img.shape)

    for i in range(0, img.shape[0], h):
        for j in range(0, img.shape[1], w):
            window = img[i:i+h, j:j+w]
            newimg[i:i+h, j:j+w] = normalize(window)

    return quantize_binary(newimg)

def separate_background(img):
    img = np.array(img, dtype=np.float64)
    newimg = np.zeros(shape=img.shape)
    newimg = normalize_by_region(img)
    return quantize_binary(newimg)


def quantize_image(img, n_colors=64):
    # Load the Summer Palace photo
    img = np.array(img, dtype=np.float64)

    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(img.shape)
    assert(d == 3)
    image_array = np.reshape(img, (w * h, d))
    print("Fitting model on a small sub-sample of the data")
    image_array_sample = shuffle(image_array)[:min(max(600, n_colors), len(image_array))]
    kmeans = cluster_colors(image_array_sample, n_colors)

    print("Predicting color indices on the full image (k-means)")
    labels = kmeans.predict(image_array)
    return kmeans, labels, w, h

from random import random
def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    image = np.zeros((w, h, 3))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    print('recreated image')
    return image


def recreate_to_image(codebook, labels, w, h):
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

def make_frame(img, n_colors):
    kmeans, labels, w, h = quantize_image(img, n_colors)
    return labels.reshape((w, h))

def limit_colors(img, n_colors):
    kmeans, labels, w, h = quantize_image(img, n_colors)
    return recreate_image(kmeans.cluster_centers_, labels, w, h), labels.reshape((w, h))
