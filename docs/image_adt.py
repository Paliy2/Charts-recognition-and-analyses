from skimage import io
import cv2 as cv
import os
from skimage import feature
import numpy as np
from random import random
from PIL import Image
from quantize import quantize
from math import inf
from sklearn.utils import shuffle
from random import random



class Image_ADT:
    def __init__(self, link='0'):
        self.img = link

    @staticmethod
    def connect_lines(lines, eps=0.01):
        lines = lines[np.argsort(lines[:, 0])]
        dist_threshold = img.shape[0] // 100
        l_start = len(lines)
        l_previous = len(lines) + 1
        while l_previous > len(lines):
            print(l_previous, len(lines))
            l_previous = len(lines)
            c = 0
            while c < len(lines):
                x1, y1, x2, y2 = lines[c]
                c2 = c + 1
                while c2 < len(lines):
                    x1, y1, x2, y2 = lines[c]
                    # k = (y1-y2)/(x1-x2)
                    # b = y2-k*x2
                    dx = x2 - x1
                    dy = y2 - y1
                    l = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

                    x1b, y1b, x2b, y2b = lines[c2]
                    # kb = (y1b-y2b)/(x1b-x2b)
                    # bb = y2b - kb*x2b
                    dxb = x2b - x1b
                    dyb = y2b - y1b
                    lb = ((x1b - x2b) ** 2 + (y1b - y2b) ** 2) ** 0.5
                    # if dx/l-eps < dxb/lb < dx/l+eps and abs(x1-x1b)+abs(x2-x2b):
                    c2 += 1
                c += 1
        return lines

    @staticmethod
    def h_filter(h_lines, close_threshold, far_threshold, close, extract_lims, eps=0.2):
        i = 0
        while i < len(h_lines):
            j = 0
            x_start, y_start, x_end, y_end = extract_lims(h_lines[i])
            L = ((x_start - x_end) ** 2 + (y_start - y_end) ** 2) ** 0.5
            while j < len(h_lines) - 1:
                Xs, Ys, Xe, Ye = extract_lims(h_lines[j])
                l = ((Xs - Xe) ** 2 + (Ys - Ye) ** 2) ** 0.5

                if Xs - x_end > close_threshold:
                    break

                try:
                    if close(x_end, Xe, close_threshold) and close(x_start, Xs, close_threshold) and \
                            (close((y_end - y_start) / L, (Ye - Ys) / l, eps) or y_end - y_start == Ye - Ys) and i != j:
                        h_lines[i] += h_lines.pop(j)
                        x_start, y_start, x_end, y_end = extract_lims(h_lines[i])
                except:
                    break
                j += 1
            i += 1

    @staticmethod
    def w_filter(h_lines, close_threshold, far_threshold, close, extract_lims, eps=0.2):
        i = 0
        while i < len(h_lines):
            j = 0
            x_start, y_start, x_end, y_end = extract_lims(h_lines[i])
            L = ((x_start - x_end) ** 2 + (y_start - y_end) ** 2) ** 0.5
            while j < len(h_lines) - 1:
                Xs, Ys, Xe, Ye = extract_lims(h_lines[j])
                l = ((Xs - Xe) ** 2 + (Ys - Ye) ** 2) ** 0.5

                if Ys - y_end > close_threshold:
                    break

                try:
                    if close(y_end, Ye, close_threshold) and close(y_start, Ys, close_threshold) and \
                            (close((x_end - x_start) / L, (Xe - Xs) / l, eps) or x_end - x_start == Ye - Ys) and i != j:
                        h_lines[i] += h_lines.pop(j)
                        x_start, y_start, x_end, y_end = extract_lims(h_lines[i])
                except:
                    break
                j += 1
            i += 1

    def simple_connector(self, img):
        '''not used at a time. can be used to reduce number of points in graph'''
        close = lambda x, i, threshold: i - threshold < x < i + threshold

        def extract_lims(line):
            current_line = np.array(line)
            xs, ys = current_line[:, 0], current_line[:, 1]
            x_start, x_end = np.min(xs), np.min(xs)
            y_start, y_end = np.max(ys), np.max(ys)
            return x_start, y_start, x_end, y_end

        def length(line):
            Xs, Ys, Xe, Ye = extract_lims(line)
            return ((Xs - Xe) ** 2 + (Ys - Ye) ** 2) ** 0.5

        h, w = img.shape[0], img.shape[1]
        h_lines = [[[inf, inf]]]
        w_lines = [[[inf, inf]]]

        close_threshold = h // 200
        far_threshold = h // 40

        i = 0
        for i in range(h):
            for j in range(w):
                if img[i][j]:
                    x = j
                    h_line = [[i, j]]
                    while x + 1 < w and img[i][x + 1]:
                        h_line.append([i, x + 1])
                        x += 1
                    if len(h_line) > 10:
                        if close(h_lines[-1][-1][0], i, close_threshold) \
                                and close(h_lines[-1][-1][1], x, far_threshold):
                            h_lines[-1] += [h_line[0], h_line[-1]]
                            # elif len(h_line) > 10:
                        else:
                            h_lines.append([h_line[0], h_line[-1]])
        h_lines.pop(0)
        self.h_filter(h_lines, close_threshold, far_threshold, close, extract_lims, 0.001)

        close_threshold = w // 200
        far_threshold = w // 40

        for j in range(w):
            for i in range(h):
                if img[i][j]:
                    w_line = []
                    y = i
                    while y + 1 < h and img[y + 1][j]:
                        w_line.append([y + 1, j])
                        y += 1
                    if len(w_line) > 2:
                        if close(w_lines[-1][-1][1], j, close_threshold) \
                                and close(w_lines[-1][-1][0], y, far_threshold):
                            w_lines[-1] += [w_line[0], w_line[-1]]
                            # elif len(w_line) > 10:
                        else:
                            w_lines.append([w_line[0], w_line[-1]])
        w_lines.pop(0)
        self.w_filter(w_lines, close_threshold, far_threshold, close, extract_lims, 0.0005)

        lines = [i for i in w_lines] + [i for i in h_lines]
        return lines

    @staticmethod
    def filter_group(data, g_count, threshold=30):
        '''remove too small groups'''
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
                        plt.show(0)
                        for index2 in neighbors[:min(50, len(neighbors))]:
                            x, y, w, k, _ = data[index1][index2]
                            if abs(x - i) < h_d_max / (index1 - index):
                                data[index1][index2][-1] = group
                                g_count[group] += 1
                                found = True
                                break
                    index1 += 1

        return self.filter_group(data, g_count, threshold), n_groups

    @staticmethod
    def remove_duplicates(graph, by_group: dict, eps):
        dist = lambda i, j, x, y: ((i - x) ** 2 + (j - y) ** 2) ** 0.5

        for c1 in range(len(graph)):
            i, j, g = graph[c1]
            by_group[g].append([i, j, g])
            for c2 in range(c1, len(graph)):
                x, y, _ = graph[c2]
                if _ != g and dist(x, y, i, j) < eps:
                    graph[c2][-1] = g
                if x - i > eps:
                    break

    def fill_gaps(self, graph, eps=5):
        by_group = {i: [] for i in set([j for j in graph[:, -1]])}
        dist = lambda i, j, x, y: ((x - i) ** 2 + (y - j) ** 2) ** 0.5
        graph = graph[np.argsort(graph[:, 0])]

        self.remove_duplicates(graph, by_group, eps)
        groups = [[i, by_group[i]] for i in by_group if by_group[i]]

        def get_lims_h(g):
            g = np.array(g)
            xs = g[:, 0]
            ys = g[:, 1]
            return np.min(xs), ys[np.argmin(xs)], np.max(xs), ys[np.argmax(xs)]

        def get_lims_w(g):
            g = np.array(g)
            ys = g[:, 0]
            xs = g[:, 1]
            return np.min(xs), ys[np.argmin(xs)], np.max(xs), ys[np.argmax(xs)]

        is_close = lambda x, y, i, j: dist(i, j, x, y) < 2 * eps
        for c in range(3):
            i = 0
            while i < len(groups):
                j = 0
                while j < len(groups) and i < len(groups):
                    g1, group1 = groups[i]
                    g2, group2 = groups[j]
                    s1x, s1y, e1x, e1y = get_lims_h(group1)
                    s2x, s2y, e2x, e2y = get_lims_h(group2)
                    if is_close(s1x, s1y, e2x, e2y):
                        # print('close')
                        groups[i][1] += groups[j][1]
                        groups.pop(j)
                    elif is_close(s2x, s2y, e1x, e1y):
                        # print('close')
                        groups[j][1] += groups[i][1]
                        groups.pop(i)
                    else:
                        s1x, s1y, e1x, e1y = get_lims_w(group1)
                        s2x, s2y, e2x, e2y = get_lims_w(group2)
                        if is_close(s1x, s1y, e2x, e2y):
                            # print('close')
                            groups[i][1] += groups[j][1]
                            groups.pop(j)
                        elif is_close(s2x, s2y, e1x, e1y):
                            # print('close')
                            groups[j][1] += groups[i][1]
                            groups.pop(i)
                    j += 1
                i += 1

        # remove_duplicates(graph, by_group, eps)

        graph = []
        for g, group in groups:
            graph += [i[:2] + [g] for i in group]

        return np.array(graph)

    def extract_graph(self, original_img, eps=4):
        w_graph = self.extract_w_graph(img)
        h_graph = self.extract_h_graph(img)
        w_graph = [[i + [original_img[int(i[1]), int(i[0])]] for i in g] for g in w_graph]
        h_graph = [[i + [original_img[int(i[1]), int(i[0])]] for i in g] for g in h_graph]

        w_graph, n_groups = self.group_w_graph(w_graph, h_d_max=img.shape[1] // 40, threshold=img.shape[1] // 20)
        h_graph, n_groups = self.group_h_graph(h_graph, n_groups=n_groups, h_d_max=img.shape[0] / 40,
                                          threshold=img.shape[0] // 20)
        graph = w_graph + h_graph
        graph = self.fill_gaps(np.array(graph), eps=(img.shape[0] + img.shape[1]) // 200)

        return graph

    def analyze(self, image, e=0.1):

        '''
        rects = pytesseract.image_to_boxes(img)
        print(rects, type(rects))
        rects = rects.split('\n')
        rects = [i.split(' ') for i in rects]
        #print(rects)
        rects = np.array([[int(_) for _ in i[1:-1]] for i in rects])
        rects = non_max_suppression(rects, probs=None)
        print(rects)
        '''

        rects = text_detection(np.array(image), min_confidence=0.6)
        img = np.uint8(quantize(image))
        plt.imshow(img)
        image = np.array(image)

        edges = img  # don't need to apply edge detection filters after quantization

        for Xs, Ys, Xe, Ye in rects:
            edges[Ys:Ye, Xs:Xe] = 0  # don't analyze text pixels for graph
            plt.plot([Xs, Xs, Xe, Xe, Xs], [Ys, Ye, Ye, Ys, Ys])

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, (img.shape[0] + img.shape[1]) // 8,
                                maxLineGap=(img.shape[0] + img.shape[1]) // 64)
        if lines is not None:
            lines = lines.reshape((len(lines), 4))
            for x1, y1, x2, y2 in lines:
                l = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if abs((y1 - y2) / l) > 0.9:
                    plt.plot([x1, x2], [y1, y2], c='blue', linewidth=1)
                elif abs((x1 - x2) / l) > 0.9:
                    plt.plot([x1, x2], [y1, y2], c='red', linewidth=1)

        plt.title('Quantized image, text and supaxes detected')
        plt.show()
        return self.extract_graph(img, np.array(image))



import matplotlib.pyplot as plt
from detect_text import detect_text as text_detection

if __name__ == '__main__':
    img = Image.open('../images/3.png')
    img_class = Image_ADT(img)
    g = img_class.analyze(img_class.img)

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

if __name__ == '__main_':
    image = io.imread('../images/12.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Generate noisy image of a square
    im = image

    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(im)
    edges2 = feature.canny(im, sigma=4)

    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)

    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=4$', fontsize=20)

    fig.tight_layout()

    # for edge in edges:
    #     print(edge)
    plt.show()
