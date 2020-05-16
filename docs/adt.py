import numpy as np
import cv2
import os
from imutils.object_detection import non_max_suppression
from sklearn.cluster import KMeans

from random import random

from sklearn.utils import shuffle
from quantize import recreate_image, limit_colors, cluster_colors, \
    normalize, separate_background, make_frame
from copy import deepcopy
from PIL import Image, ImageEnhance, ImageFilter

import matplotlib.pyplot as plt


def total_processing(path, n_colors=3):
    img = Image.open(path)
    width, height = img.width, img.height
    print(width, height)
    if width > 400:
        width, height = 400, int(height * 400 / width)
        img.thumbnail((width, height), Image.ANTIALIAS)

    converter = ImageEnhance.Color(img)
    s = max(3, (img.width + img.height) // 400)
    contrast = ImageEnhance.Contrast(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=s))
    img = contrast.enhance(1)

    image = deepcopy(img)
    image = np.array(image)
    width, height = image.shape[0], image.shape[1]
    print(width, height)
    edges = np.uint8(separate_background(np.array(img)))

    original_image = np.array(converter.enhance(8))

    for i in range(width):
        for j in range(height):
            if edges[i][j] == 0:
                original_image[i][j] = 0

    # filter_image = deepcopy(original_image)
    original_image, frame = limit_colors(original_image, n_colors)
    # frame = frame + 1
    # filter_frame = make_frame(filter_image, 64)
    # for c in range(64):
    #    points = [[i, j] for i in range(width) for j in range(height) if filter_frame[i][j]==c]
    #    if points:
    #        points = np.array(points)
    #        area = (np.max(points[:, 0])-np.min(points[:, 0]))*\
    #            (np.max(points[:, 1])-np.min(points[:, 1]))
    #        if len(points)/area < 1/(128+64):
    #            for i, j in points:
    #                frame[i][j] = 0

    plt.imshow(frame)
    plt.show()
    # frame = frame + 1
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] == 0:
                frame[i][j] = 0

    plt.imshow(frame)
    plt.show()

    graphs = extract_graphs(frame, frame.shape[0] * frame.shape[1] // 8000)
    return graphs


def fill(img, start_point, color):
    img = deepcopy(img)
    points = [start_point]
    width, height = img.shape
    x, y = start_point

    y_ = y + 1
    while y_ < height - 1 and img[x][y_] == color:
        points.append([x, y_])
        y_ += 1

    y_ = y - 1
    while y_ > -1 and img[x][y_] == color:
        points.append([x, y_])
        y_ -= 1

    x_ = x + 1
    while x_ < width - 1 and img[x_][y] == color:
        points.append([x_, y])
        x_ += 1

    x_ = x - 1
    while x_ > -1 and img[x_][y] == color:
        points.append([x_, y])
        x_ -= 1

    for i, j in points:
        img[i][j] = -1

    if len(points) == 1:
        to_return = []
        if 0 < x < width - 2 and 0 < y < height - 2:
            for i, j in [[x + 1, y + 1], [x + 1, y - 1], [x - 1, y + 1], [x - 1, y - 1]]:
                to_return.append([i, j])
        return to_return, img

    c = 1
    for p in points[1:]:
        new_points, img = fill(img, p, color)
        points += new_points
        c += 1
    return points, img


def extract_graphs(frame, threshold):
    all_points = [[i, j] for i in range(frame.shape[0]) for j in range(frame.shape[1]) if frame[i][j]]
    graphs = []
    while all_points:
        x, y = all_points.pop(0)
        points, _ = fill(frame, [x, y], frame[x][y])
        graphs.append(points)
        all_points = list(filter(lambda x: x not in points, all_points))
    graphs = list(filter(lambda x: len(x) > threshold, graphs))
    return [np.array(g) for g in graphs]


def plot_graphs(graphs):
    l = sum([len(g) for g in graphs]) / len(graphs)
    for g in graphs:
        c = [random(), random(), random()]
        plt.scatter(g[:, 1], g[:, 0], color=c)
    plt.show()


class Resizer:
    def __init__(self, img):
        self.original_img = deepcopy(img)

    def resize(self):
        return


if __name__ == '__main__':
    n_clusters = [7, 2, 2, 3, 7, 3, 5, 5, 5]
    # for i, n in zip(['../images/pencil.jpg']):
    #     print(i)
    g = total_processing('../images/pencil.jpg', 3)
    plot_graphs(g)