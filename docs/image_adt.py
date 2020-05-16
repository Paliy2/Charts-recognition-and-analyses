from skimage import io
import cv2 as cv
import os
from skimage import feature
import numpy as np
from random import random
from PIL import Image
from quantize import separate_background
from math import inf
from sklearn.utils import shuffle
from random import random



class Image_ADT:
    def __init__(self, link='0'):
        self.img = link

    @staticmethod
    def filter_group(data, g_count, threshold=5):
        '''remove unnecessary data from groups'''
        return [i[:2] + [i[-1]] for g in data for i in g]

    @staticmethod
    def extract_w_graph(img):
        graph = []
        n = 0
        for j in range(img.shape[1]):
            graph.append([])
            bar = img[:, j]
            y_start = 0
            started_line = False
            for i in range(img.shape[0]):
                if started_line:
                    if bar[i] == 0:
                        graph[-1].append([j, (i + y_start) // 2, i - y_start])
                        started_line = False

                else:  # not started_line
                    if bar[i] != 0:  # != 0
                        y_start = i
                        started_line = True
        return graph

    @staticmethod
    def extract_h_graph(img):
        graph = []
        for j in range(img.shape[0]):
            graph.append([])
            bar = img[j]
            y_start = 0
            started_line = False
            for i in range(img.shape[1]):
                if started_line:
                    if bar[i] == 0:
                        graph[-1].append([(i + y_start) // 2, j, i - y_start])
                        started_line = False

                else:  # not started_line
                    if bar[i] != 0:  # != 0
                        y_start = i
                        started_line = True
        return graph

    def group_w_graph(self, graph, threshold=30, n_groups=0, h_d_max=10):
        g_count = {}
        data = [[[_ for _ in j] + [-1] for j in i] for i in graph]
        for index in range(len(data) - 1):
            for i in range(len(data[index])):
                if data[index][i][-1] == -1:
                    data[index][i][-1] = n_groups
                    g_count[n_groups] = 0
                    n_groups += 1
            if data[index] != []:
                for i, j, h, c, group in data[index]:
                    index1 = index + 1
                    found = False
                    while index1 < index + 5 and index1 < len(data) and not found:
                        if data[index1] != []:
                            dists = [abs(_[1] - j) + np.sum(abs(c - _[-2])) for _ in data[index1]]
                            neighbors = np.argsort(dists)
                            # index2 = np.argmin([abs(_[1]-j)+np.sum(abs(c-_[-2])) for _ in data[index1]])
                            for index2 in neighbors[:min(50, len(neighbors))]:
                                x, y, w, k, _ = data[index1][index2]
                                if abs(y - j) < h_d_max:  # /(index1-index):
                                    data[index1][index2][-1] = group
                                    g_count[group] += 1
                                    found = True
                                    break
                        index1 += 1

        return self.filter_group(data, g_count, threshold), n_groups

    def group_h_graph(self, graph, threshold=30, n_groups=0, h_d_max=10):
        g_count = {}
        data = [[[_ for _ in j] + [-1] for j in i] for i in graph]
        for index in range(len(data) - 1):
            for i in range(len(data[index])):
                if data[index][i][-1] == -1:
                    data[index][i][-1] = n_groups
                    g_count[n_groups] = 0
                    n_groups += 1
            for i, j, h, c, group in data[index]:
                index1 = index + 1
                found = False
                while index1 < index + 5 and index1 < len(data) and not found:
                    if data[index1] != []:
                        dists = [abs(_[0] - i) + np.sum(abs(c - _[-2])) for _ in data[index1]]
                        neighbors = np.argsort(dists)
                        # plt.show()
                        plt.clf()
                        for index2 in neighbors[:min(50, len(neighbors))]:
                            x, y, w, k, _ = data[index1][index2]
                            if abs(x - i) < h_d_max / (index1 - index):
                                data[index1][index2][-1] = group
                                g_count[group] += 1
                                found = True
                                break
                    index1 += 1

        return self.filter_group(data, g_count, threshold), n_groups

    def extract_graph(self, img, original_img, eps=4):
        w_graph = self.extract_w_graph(img)
        h_graph = self.extract_h_graph(img)
        w_graph = [[i + [original_img[int(i[1]), int(i[0])]] for i in g] for g in w_graph]
        h_graph = [[i + [original_img[int(i[1]), int(i[0])]] for i in g] for g in h_graph]

        w_graph, n_groups = self.group_w_graph(w_graph, h_d_max=img.shape[1] // 40, threshold=img.shape[1] // 20)
        h_graph, n_groups = self.group_h_graph(h_graph, n_groups=n_groups, h_d_max=img.shape[0] / 40,
                                          threshold=img.shape[0] // 20)
        graph = w_graph + h_graph
        return graph

    def analyze(self, image, e=0.1):
        rects = text_detection(np.array(image), min_confidence=0.6)
        plt.imshow(image)
        img = np.array(image)

        edges = separate_background(img)  # don't need to apply edge detection filters after quantization
        plt.imshow(edges)
        # plt.show()

        for Xs, Ys, Xe, Ye in rects:
            edges[Ys:Ye, Xs:Xe] = 0  # don't analyze text pixels for graph
            plt.plot([Xs, Xs, Xe, Xe, Xs], [Ys, Ye, Ye, Ys, Ys])

        lines = cv.HoughLinesP(np.uint8(edges*255), 1, np.pi / 180, \
            (img.shape[0] + img.shape[1]) // 8, maxLineGap=(img.shape[0] + img.shape[1]) // 64)
        if lines is not None:
            lines = lines.reshape((len(lines), 4))
            for x1, y1, x2, y2 in lines:
                l = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if abs((y1 - y2) / l) > 0.9:
                    plt.plot([x1, x2], [y1, y2], c='blue', linewidth=1)
                elif abs((x1 - x2) / l) > 0.9:
                    plt.plot([x1, x2], [y1, y2], c='red', linewidth=1)

        plt.title('Quantized image, text and supaxes detected')
        # plt.show()
        plt.clf()
        return self.extract_graph(edges, np.array(image))



import matplotlib.pyplot as plt
from detect_text import detect_text as text_detection
# from images import images
if __name__ == '__main__':
    img = Image.open('../images/pencil.jpg')
    img_class = Image_ADT(img)
    g = img_class.analyze(img)
    # plt.imshow(g)
    # plt.show()
    i = np.array(g)
    colors = {i: [random(), random(), random()] for i in set(i[:, -1])}
    print('initial number of clusters', len(colors))
    n_threshold = 100  # minimal number of points required in a cluster
    by_group = {i: [[x, y] for x, y, g in g if g == i] for i in set(i[:, -1]) if
                len([[x, y] for x, y, g in g if g == i]) > n_threshold}
    print('Final number of clusters', len(by_group))
    plt.imshow(img_class.img)
    for x, y in by_group.items():
        plt.scatter([i[0] for i in y], [i[1] for i in y], color=colors[x])
    plt.show()


# Nosiy filters explained
# if __name__ == '__main_':
#     image = io.imread('../images/plane.jpg')
#     image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#     # Generate noisy image of a square
#     im = image
#
#     # Compute the Canny filter for two values of sigma
#     edges1 = feature.canny(im)
#     edges2 = feature.canny(im, sigma=4)
#
#     # display results
#     fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
#                                         sharex=True, sharey=True)
#
#     ax1.imshow(im, cmap=plt.cm.gray)
#     ax1.axis('off')
#     ax1.set_title('noisy image', fontsize=20)
#
#     ax2.imshow(edges1, cmap=plt.cm.gray)
#     ax2.axis('off')
#     ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)
#
#     ax3.imshow(edges2, cmap=plt.cm.gray)
#     ax3.axis('off')
#     ax3.set_title('Canny filter, $\sigma=4$', fontsize=20)
#
#     fig.tight_layout()
#
#     # for edge in edges:
#     #     print(edge)
#     plt.show()
#