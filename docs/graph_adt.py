import numpy as np
import matplotlib.pylab as plt
from docs.this_dict import Dict
from docs.arrays import Array2D


class GraphADT:
    """A class to represent digital graph or chart"""

    def __init__(self, graph=None, read_name=None):
        """
        Parameters:
        points: numpy array
            points of an image to work with
        """
        if read_name and read_name.endswith('.npy'):
            self.graph = self._read_npy(read_name)
        else:
            self.graph = graph
        self.img = self.graph
        self.graph = self.convert_to_2d()
        # testing graph points
        self.points = self._convert_to_x_y()

    def __str__(self):
        """print all points"""
        res = ''
        res += f'<object: GraphAdt, len: {len(list(self.keys))}>'
        return res

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return type(self) == type(other) and len(self.keys) == len(other.values)

    def __getitem__(self, item):
        return self.graph[item]

    def convert_to_2d(self):
        """Transfer numpy 2d array into data structure Array2D to make it work faster"""
        if isinstance(self.graph, Array2D):
            return

        array2d = Array2D(len(self.graph), len(self.graph[0]))
        for y in range(len(self.graph)):
            for x in range(len(self.graph[0])):
                array2d[y, x] = self.graph[y][x]
        return array2d

    def _convert_to_x_y(self):
        """
        :param self.graph: numpy array
        return: dict
            dict of items like {x: y}
        """
        y_len = self.graph.num_rows()
        x_len = self.graph.num_cols()

        res = []
        y_len_02 = y_len * 0.2
        for i in range(x_len):
            x = []
            # add row values
            x += [(y + 1) * int(self.graph[y, i]) for y in range(y_len)]
            # last_y = 0
            # try:
            #     col_sum = [int(self.graph[j, i-1]) for j in range(y_len)]
            # except:
            #     col_sum = [0]
            # col_sum = sum(col_sum)
            # for y in range(y_len):
            #     # if 0 <= col_sum <= y_len * 0.01:
            #     #     x.append(last_y)
            #     if y - last_y <= y_len_02:
            #         x.append((y + 1) * int(self.graph[y, i]))
            #     if int(self.graph[y, i]) == 1:
            #         last_y = y
            #
            x = self.median(x)
            res.append(x)  # it's height
        # we have res with all y values
        self.points = Dict(enumerate(res))
        points = sorted(self.points.items())  # sorted by key, return a list of tuple
        x, y = zip(*points)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.show()
        return Dict(enumerate(res))

    def show(self):
        """Show an image of graph on screen"""
        plt.imshow(self.img)
        plt.savefig('show_res')
        plt.show()

    def plot(self):
        points = sorted(self.points.items())  # sorted by key, return a list of tuple
        x, y = zip(*points)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        # plt.plot(self.points[:, 0], self.points[:, 1])

    def save_plot(self):
        """
        Plot an image of the graph
        self.dict is alwys dict of two itemx x: y"""
        self.plot()
        plt.savefig('saved_graph.png', bbox_inches='tight')
        plt.clf()
        # plt.plot(x, y)

    def to_csv(self, f_name='points.csv'):
        '''save point x and'''
        with open(f_name, 'w', encoding='utf-8') as f:
            for x, y in zip(self.keys, self.values):
                f.write(str(x) + ';' + str(y) + '\n')

    def _read_npy(self, file='data.npy'):
        """read numpy data from txt file"""
        res = np.load(file, allow_pickle=True)
        self.graph = res
        return res

    def reveal(self, axes=True):
        """
        Create a short exact prediction
        for next 20% that can br in the graph

        : param axes: bool
            False to turn axes off
        : param save: bool
            True ti save data to file
        """
        p_y = list(self.values)
        p_x = list(self.keys)

        mymodel = np.poly1d(np.polyfit(p_x, p_y, 7))  # 15 is good
        # predict behaviour in current decade
        myline = np.linspace(int(min(p_x)), int(max(p_x) * 1.2), min(p_y), max(p_y))
        if not axes:
            plt.axis('off')
        plt.plot(p_x, p_y, linewidth=5)
        plt.plot(myline, mymodel(myline), linewidth=5)

        plt.savefig('prediction.png', bbox_inches='tight')
        print('Saved reveal to prediction.png!')
        plt.clf()
        # plt.show()

    def simple_reveal(self):
        """Make linear regression of given graph"""
        a, b = self.get_a_b_linear_regr()
        X = [0, max(self.keys) * 1.05]
        Y = [a, a + b * X[1]]
        plt.plot(X, Y)
        self.plot()
        plt.savefig('prediction.png', bbox_inches='tight')
        print('Regression saved to prediction.png!')
        plt.clf()

    def get_a_b_linear_regr(self):
        """
        Get data to create a linear regression
        """
        p_x = self.keys
        p_y = self.values

        x_2 = 0
        for i in p_x:
            x_2 += i ** 2
        x, y, n = sum(p_x), sum(p_y), len(p_x)
        xy = 0
        for a, b in zip(p_x, p_y):
            xy += a * b
        try:
            b = (n * xy - x * y) / (n * x_2 - x ** 2)
            a = (y * x_2 - x * xy) / (n * x_2 - x ** 2)
        except ZeroDivisionError:
            return 0, 0
        return a, b

    @staticmethod
    def median(lst):
        """Get the floor of medium values in list"""
        res = 0
        for x in lst:
            res += x
        return res // len(lst)

    @property
    def keys(self):
        return self.points.keys()

    @property
    def values(self):
        return self.points.values()


class Multigraph():
    def __init__(self, graphs):
        self.graphs = [GraphADT(g) for g in graphs]

    def show(self, i=None):
        if i == None:
            for g in self.graphs:
                g.plot()
        elif 0 < i < len(self.graphs):
            self.graphs[i].plot()
        plt.show()

    def save(self):
        pass