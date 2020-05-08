import numpy as np
import matplotlib.pylab as plt


class GraphADT:
    def __init__(self, graph):
        """
        Parameters:
        points: numpy array
            points of an image to work with
        """
        self.graph = graph
        # testing graph points
        self.points = self.convert_to_x_y()
        # self.points[4] = 50

    def __str__(self):
        """print all points"""
        res = ''
        res += f'<object: GraphAdt, len: {len(self.points)}>'
        return res

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(self, other):
            p1 = self.points
            p2 = other.points
            return p1 == p2
        return False

    def __getitem__(self, item):
        return self.graph[item]

    def convert_to_x_y(self):
        """
        :param self.graph: numpy array
        return: dict
            dict of items like {x: y}
        """
        y_len = len(self.graph)
        x_len = len(self.graph[0])
        res = []
        for i in range(x_len):
            x = []
            # add row values
            x += [(y + 1) * int(self.graph[y][i]) for y in range(y_len)]
            x = self.median(x)
            res.append(x)  # it's height
        # we have res with all y values
        self.points = dict(enumerate(res))
        return dict(enumerate(res))

    def show(self):
        """Show an image of graph on screen"""
        plt.imshow(self.graph)
        plt.show()

    def plot(self, save=False):
        """
        Plot an image of the graph
        self.dict is alwys dict of two itemx x: y"""

        points = sorted(self.points.items())  # sorted by key, return a list of tuple
        x, y = zip(*points)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        if save:
            plt.savefig('figure')
        plt.show()

    def to_csv(self):
        '''save point x and'''
        with open('points.csv', 'w', encoding='utf-8') as f:
            for x, y in zip(self.points.keys(), self.points.values()):
                f.write(x + ';' + y + '\n')

    def save(self, file='data.npy'):
        '''save total graph'''
        # with open('array.txt', 'w', encoding='utf-8') as f:
        #     for slice_2d in self.graph:
        #         np.savetxt(f, slice_2d)
        np.save(file, self.graph)

    def _read_npy(self, file='data.npy'):
        """read numpy data from txt file"""
        # with open(file, 'r', encoding='utf-8') as f:
        #     res = np.loadtxt(f)
        # self.graph = res
        res = np.load(file, allow_pickle=True)
        self.graph = res
        return res

    def reveal(self, axes=True, save=False):
        """
        Create a short exact prediction
        for next 20% that can br in the graph

        : param axes: bool
            False to turn axes off
        : param save: bool
            True ti save data to file
        """
        p_y = list(self.points.values())
        p_x = list(self.points.keys())

        mymodel = np.poly1d(np.polyfit(p_x, p_y, 3))  # 15 is good
        # predict behaviour in current decade
        myline = np.linspace(int(max(p_x) * .8), int(max(p_x) * 1.2), min(p_y), max(p_y))
        if not axes:
            plt.axis('off')
        if save:
            plt.savefig('reveal')
        plt.plot(myline, mymodel(myline), linewidth=5)
        plt.plot(p_x, p_y, linewidth=5)

        plt.show()

    def simple_reveal(self, save=False):
        """Make linear regression of given graph"""
        a, b = self.get_a_b_linear_regr()
        X = [0, max(self.points.keys()) * 1.05]
        Y = [a, a + b * X[1]]
        plt.plot(X, Y)
        self.plot()
        if save:
            plt.savefig('regression')
        plt.show()

    def get_a_b_linear_regr(self):
        """
        Get data to create a linear regression
        """
        p_x = self.points.keys()
        p_y = self.points.values()

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
        res = 0
        for x in lst:
            res += x
        return res // len(lst)


if __name__ == '__main__':
    data = np.load('data.npy')
    graph = GraphADT(data)
    graph.convert_to_x_y()
    # graph.plot(save=True)
    # print('Plotted')
    # graph.simple_reveal()
    graph.reveal()
    # s = graph._read_npy()
    # graph.show()

# Files::
# regression - plot with linear regression
# figure -- current x and y
# reveal -- polynomial regression
