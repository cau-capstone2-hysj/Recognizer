import unittest

import numpy as np
from components.landmark.mylandmark import MyLandmark


class MyLandmarkTest(unittest.TestCase):
    def test_visualize_1(self):
        value = MyLandmark((1, 2, 3), 0.5)
        self.assertEqual(value.vis, 0.5)

    def test_visualize_2(self):
        value = MyLandmark([1, 2, 3], 0.5)
        self.assertEqual(value.y, 2)
