import unittest
import numpy as np
from graph_adt import GraphADT
from arrays import Array2D
import os


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.data = np.load('data.npy')
        self.short_data = [[1, 0], [0, 1], [1, 1]]
        self.D2data = Array2D(3, 2)
        for y in range(3):
            for x in range(2):
                self.D2data[y, x] = self.short_data[y][x]
        self.xy_dict = dict()
        self.xy_dict[0] = 1
        self.xy_dict[1] = 1
        self.graph = GraphADT(self.data)
        self.short_graph = GraphADT(self.short_data)

    def test_read_npy(self):
        # self.assertIsInstance(self.data, self.graph._read_npy('data.npy'))
        self.assertEqual(self.data.all(), self.graph._read_npy('data.npy').all())
        # self.assertEqual(self.graph.graph.num_cols(), GraphADT(read_name='data.npy').graph.num_cols())
        # self.assertEqual(self.graph.graph.num_rows(), GraphADT(read_name='data.npy').graph.num_rows())


    def test_to_2d(self):
        self.assertEqual(self.short_graph.graph.num_rows(), self.D2data.num_rows())
        self.assertEqual(self.short_graph.graph.num_cols(), self.D2data.num_cols())

        self.assertIsInstance(self.graph.graph, Array2D)

    def test_to_x_y(self):
        self.assertIsInstance(self.graph.points, dict)
        self.assertEqual(self.short_graph._convert_to_x_y(), self.xy_dict)

    def test_a_b_linear_regr(self):
        self.assertEqual(self.short_graph.get_a_b_linear_regr(), (1.0, 0.0))
        # self.assertRaises(self.short_graph.get_a_b_linear_regr(), ZeroDivisionError)

    def test_plot_saving(self):
        self.graph.plot(save=True)
        self.graph.reveal(axes=False, save=True)
        self.graph.simple_reveal(save=False)
        self.assertIn('figure.png', os.listdir())
        self.assertIn('reveal.png', os.listdir())
        self.assertNotIn('regression', os.listdir())

    def test_data_saving(self):
        # self.graph.save(file='test.npy')
        self.graph.to_csv(f_name='points.csv')
        self.assertIn('points.csv', os.listdir())
        self.assertNotIn('tests.npy', os.listdir())
        os.remove('points.csv')
        # os.remove('test.npy')

    def test_str(self):
        self.assertEqual(str(self.short_graph), "<object: GraphAdt, len: 2>")
        self.assertEqual(str(self.graph), "<object: GraphAdt, len: 225>")
        self.assertEqual(str(self.short_graph), repr(self.short_graph))

    def test_eq(self):
        self.assertEqual(self.graph, self.graph)
        self.assertNotEqual(self.graph, self.short_graph)

    def test_dict(self):
        self.assertEqual(len(self.graph.points.keys()), len(self.graph.keys))
        self.assertEqual(len(self.graph.points.values()), len(self.graph.values))
        self.assertEqual(len(self.short_graph.keys), 2)
        self.assertEqual(len(self.short_graph.values), 2)

    def tests_median(self):
        self.assertEqual(self.graph.median([2, 0]), 1)
        self.assertEqual(self.graph.median([0, 1]), 0)
        self.assertEqual(self.graph.median([3, 0, 1]), 1)
        self.assertEqual(self.graph.median([2, 1]), 1)


if __name__ == '__main__':
    unittest.main()
