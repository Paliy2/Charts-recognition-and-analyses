import numpy as np
from sklearn.cluster import KMeans
from quantize import recreate_image, limit_colors, cluster_colors, \
    normalize, separate_background, make_frame
from copy import deepcopy
from PIL import Image, ImageEnhance, ImageFilter

import matplotlib.pyplot as plt


def get_start_end(cluster: list):
    '''
    Get starting and ending points of a cluster of pixels
    Returns:
        start (np.array): [min_x, min_y]
            min_x - minimal x of the points
            min_y - average value of y for points at min_x
        end (np.array): [max_x, max_y]
            max_x - maximal x of the points
            max_y - average value of y for points at max_x
       h_start (float): height* of the cluster at min_x
       h_end   (float): height* of the cluster at max_x
                        
       *(difference between max and min y's at that point)

    
    8
    7  00000______________________> y_min = (6+7)/2
    6  000000000             h_min = 2
    5  |   000000000         (2 points with the same x=x_min)
    4  |      0000000000
    3  |           000000000
    2  |                0000000 --> y_max = 2
    1  |                      |     h_max=1
    0  x_min                  x_max

    0000000001111111111222222222
    1234567890123456789012345678
    '''

    cluster = np.array(cluster)
    min_x = np.min(cluster[:, 0])
    min_ys = [j for i, j in cluster if i==min_x]
    max_x = np.max(cluster[:, 0])
    max_ys = [j for i, j in cluster if i==max_x]
    start = min_x, np.mean(min_ys)
    end = max_x, np.mean(max_ys)
    h_start, h_end = max(min_ys)-min(min_ys), max(max_ys)-min(max_ys)
    return np.array(start), np.array(end), h_start, h_end


def find_neighbors(img: np.array, start_point: tuple, color: int)->list:
    '''for flood-fill instrument
    Args:
        img (np.array): 2D array of integers
        start_point (tuple): point to fill from
        color: color we need to look for

    Returns:
        points (list): all points in the north\south\east\west directions
        from the give one that have the color
    '''

    width, height = img.shape
    x, y = start_point
    points = []

    y_ = y+1
    while y_ < height-1 and img[x][y_]:
        points.append((x, y_))
        y_ += 1

    y_ = y-1
    while y_ > -1 and img[x][y_]:
        points.append((x, y_))
        y_ -= 1

    x_ = x+1
    while x_ < width-1 and img[x_][y]:
        points.append((x_, y))
        x_ += 1

    x_ = x-1
    while x_ > -1 and img[x_][y]:
        points.append((x_, y))
        x_ -= 1
    return points


def fill(frame: np.array, start_point: tuple) -> list:
    '''flood-fill instrument
    Args:
        frame (np.array): 2D array of integers
        start_point (tuple): point to fill from

    Returns:
        analyzed_points (list): all points around the given one
                                with the same color
    '''

    color = frame[start_point]
    img = frame==color
    img[start_point] = 0
    analyzed_points = []
    points_to_analyze = [start_point]
    while points_to_analyze:
        l = len(points_to_analyze)
        for i in range(l):
            p = points_to_analyze.pop(0)
            new_points = find_neighbors(img, p, color)
            for i, j in new_points:
                img[i][j] = 0
            points_to_analyze += new_points
            analyzed_points.append(p)
            img[p[0]][p[1]] = 0
    return analyzed_points


class ImageADT():
    def __init__(self, path_to_image):
        self.dir = path_to_image
        self.img = Image.open(path_to_image)

    @staticmethod
    def increase_saturation(img, n):
        '''Increase image saturation
        Args:
            img: an image opened with PIL Image
            n (float): intensity of filter

        Returns:
            modified img
        '''

        return ImageEnhance.Color(img).enhance(n)

    @staticmethod
    def increase_contrast(img, n):
        '''Increase image contrast
        Args:
            img: an image opened with PIL Image
            n (float): intensity of filter

        Returns:
            modified img
        '''

        return ImageEnhance.Contrast(img).enhance(n)

    @staticmethod
    def resize_to(img, width : int, height: int):
        '''Resize image tp size (width, height)
        Args:
            img: an image opened with PIL Image
            width (int): width
            height (int): height

        Returns:
            resized img
        '''

        img.thumbnail((width, height), Image.ANTIALIAS)
        return img

    @staticmethod
    def scale_image(img, max_width=500):
        '''Scale image, so that it's width is < max_width
        Args:
            img: an image opened with PIL Image
            max_width (int): maximum possible width

        Returns:
            properly resized img
        '''

        width, height = img.width, img.height
        if img.width > max_width:
            width, height = max_width, int(height*max_width/width)
            ImageADT.resize_to(img, width, height)
        return img, (width, height)

    @staticmethod
    def blur_image(img, s=1):
        '''Blur image
        Args:
            img: an image opened with PIL Image
            s (float): size of kernel for Gaussian filter

        Returns:
            modified img
        '''

        img = img.filter(ImageFilter.GaussianBlur(radius=s))
        return img

    def analyze(self, n_colors=3):
        '''Extract graphs from an image'''

        img = deepcopy(self.img)
        img, (width, height) = self.scale_image(img)
        #width, height = img.width, img.height
        img = self.blur_image(img, max(3, (width+height)//400))
        img = self.increase_contrast(img, 1)

        edges = np.uint8(separate_background(np.array(img)))
        coolorful_image = np.array(self.increase_contrast(img, 8))

        width, height = edges.shape[0], edges.shape[1]

        for i in range(width):
            for j in range(height):
                if edges[i][j] == 0:
                    coolorful_image[i][j] = 0

        coolorful_image, frame = limit_colors(coolorful_image, n_colors)

        frame = frame + 1
        for i in range(width):
            for j in range(height):
                if edges[i][j] == 0:
                    frame[i][j] = 0
        plt.imshow(edges)
        plt.savefig('edges.png', asex=False)
        plt.clf()

        graphs = ImageADT.extract_graphs(frame, np.sum(edges)/30)
        self.plot_graphs(graphs, False)
        return [self.to_form(g, width, height) for g in graphs]

    @staticmethod
    def extract_graphs(frame, threshold):
        '''Find isolated regions of pixels of the same class on the image

        1) Applies flood-fill instrument to non-background pixels
           to find regions of the same class
        2) Connect regions found above based on their parameters
        3) Return a list of graphs
        '''

        # thrreshold to determine whether clusters are the same graph
        dist_threshold = np.array([frame.shape[0], frame.shape[1]])/100
        eps = frame.shape[1]/80

        by_color = {}
        # non-background points
        all_points = [(i, j) for i in range(frame.shape[0])\
                      for j in range(frame.shape[1]) if frame[i][j]]


        # too small blobs of color are considered noise
        area_threshold = frame.shape[0]*frame.shape[1]//8000
       

        # 1) Apply flood-fill instrument to find regions of the same class
        while all_points:
            x, y = all_points.pop(0)
            if frame[x][y] not in by_color:
                by_color[frame[x][y]] = []
            points = fill(frame, (x, y))

            if len(points) > area_threshold:
                by_color[frame[x][y]].append(points)

            all_points = list(filter(lambda x: not (x in points), all_points))


        # 2) Connect regions found above based on their parameters
        previous_len = sum([len(by_color[i]) for i in by_color])+1

        while previous_len > sum([len(by_color[i]) for i in by_color]):
            previous_len = sum([len(by_color[i]) for i in by_color])
            for color in by_color:
                i = 0
                while i < len(by_color[color]):
                    j = i+1
                    while j < len(by_color[color]):
                        sa, ea, hsa, hea = get_start_end(by_color[color][i])
                        sb, eb, hsb, heb = get_start_end(by_color[color][j])
                        if all(i<j for i, j in zip(abs(ea-sb), dist_threshold)) and hsb-eps < hea < hsb+eps:
                            by_color[color][i] += by_color[color][j]
                            by_color[color].pop(j)

                        if all(i<j for i, j in zip(abs(eb-sa), dist_threshold)) and hsa-eps < heb < hsa+eps:
                            by_color[color][j] += by_color[color][i]
                            by_color[color].pop(i)
                        j += 1
                    i += 1

        # 3) Combine results and return
        graphs = []
        for color in by_color:
            graphs += by_color[color]

        # filter assumed "graphs" by the number of points given by user
        graphs = list(filter(lambda x: len(x)>threshold, graphs))
        return graphs

    @staticmethod
    def to_form(graph, width, height):
        '''Turn a graph into a form of 0 and 1,
           0 - no points, 1 - point of graph

        Args:
            graph (np.array): pixels of a graph
            w (int): width of the image
            h (int): height of the image
        Returns:
            form (np.array): form of 0 and 1
        '''

        form = np.zeros((width, height))
        for i, j in graph:
            form[i][j] = 1
        return form

    @staticmethod
    def plot_graphs(graphs, show=False):
        '''plot all graphs and save\show
        Args:
            graphs (list of np.arrays): pixels identified
            as individual graphs
            show (bool): show the image if True, save otherwise
        '''

        for i in range(len(graphs)):
            g = np.array(graphs[i])
            plt.scatter(g[:, 1], g[:, 0])
        if show:
            plt.show()
        else:
            #!!!! Заміни на нормальний шлях, який тобі треба
            plt.savefig('color_blobs.png')
            plt.clf()



if __name__ == '__main__':
    from graph_adt import Multigraph, GraphADT

    img = ImageADT('../edges.png')

    graphs = img.analyze(2)

    GraphADT(graphs[-1]).plot()
    mgraph = Multigraph(graphs)
    mgraph.show()
