import unittest

import numpy as np
from components.landmark.mylandmark import MyLandmark
from components.landmark.vector3d import Vector3d


class MyLandmarkTest(unittest.TestCase):
    def test_visualize_1(self):
        value = MyLandmark((1, 2, 3), 0.5)
        self.assertEqual(value.vis, 0.5)

    def test_visualize_2(self):
        value = MyLandmark([1, 2, 3], 0.5)
        self.assertEqual(value.y, 2)

    def test_coord_xyz_1(self):
        value = MyLandmark((1, 2, 3), 0.5)
        self.assertEqual(value.coord.x, 1)
        self.assertEqual(value.coord.y, 2)
        self.assertEqual(value.coord.z, 3)

    def test_coord_xyz_2(self):
        value = MyLandmark([1, 2, 3], 0.5)
        coord = Vector3d(value.coord)
        self.assertEqual(coord.x, 1)
        self.assertEqual(coord.y, 2)
        self.assertEqual(coord.z, 3)
