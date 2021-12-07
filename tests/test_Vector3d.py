import unittest

import numpy as np
from components.landmark.vector3d import Vector3d


class Vector3dTest(unittest.TestCase):
    def test_equality0(self):
        value1 = Vector3d([1, 2, 3])
        value2 = Vector3d(1, 2, 3)
        self.assertEqual(value1, value2)

    def test_initialize(self):
        value1 = Vector3d(1, 2, 3)
        value2 = Vector3d(np.array([1, 2, 3]))
        self.assertEqual(value1, value2)

    def test_initialize_2args(self):
        value1 = Vector3d(1, 2)
        value2 = Vector3d(1, 2, 0)
        self.assertEqual(value1, value2)

    def test_malformed_vector(self):
        with self.assertRaises(ValueError):
            Vector3d([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            Vector3d([1])
        with self.assertRaises(ValueError):
            Vector3d(1, 2, 3, 4)
        with self.assertRaises(TypeError):
            Vector3d(1)

    def test_unit_vector0(self):
        value1 = Vector3d(1, 2, 3)
        self.assertEqual(
            value1.unitvector(),
            Vector3d(1 / (14 ** 0.5), 2 / (14 ** 0.5), 3 / (14 ** 0.5)),
        )

    def test_angle_0(self):
        value1 = Vector3d(1, 0, 0)
        value2 = Vector3d(1, 0, 0)
        np.testing.assert_almost_equal(value1.anglebtw(value2), 0.0)

    def test_angle_45(self):
        value1 = Vector3d(1, 0, 0)
        value2 = Vector3d(1, 1, 0)
        np.testing.assert_almost_equal(value1.anglebtw(value2), 45.0)

    def test_angle_60(self):
        value1 = Vector3d(0, 0, 1)
        value2 = Vector3d(0, 3 ** 0.5, 1)
        np.testing.assert_almost_equal(value1.anglebtw(value2), 60.0)

    def test_angle_90(self):
        value1 = Vector3d(1, 0, 0)
        value2 = Vector3d(0, 1, 0)
        np.testing.assert_almost_equal(value1.anglebtw(value2), 90.0)

    def test_angle_180(self):
        value1 = Vector3d(1, 0, 0)
        value2 = Vector3d(-1, 0, 0)
        np.testing.assert_almost_equal(value1.anglebtw(value2), 180.0)

    def test_dist_0(self):
        value1 = Vector3d(1, 0, 0)
        value2 = Vector3d(1, 0, 0)
        np.testing.assert_almost_equal(value1.distbtw(value2), 0.0)

    def test_dist_1(self):
        value1 = Vector3d(1, 0, 0)
        value2 = Vector3d(2, 0, 0)
        np.testing.assert_almost_equal(value1.distbtw(value2), 1.0)

    def test_dist_2(self):
        value1 = Vector3d(1, 1, 0)
        value2 = Vector3d(2, 2, 0)
        np.testing.assert_almost_equal(value1.distbtw(value2), 2 ** 0.5)

    def test_dist_sqrt2(self):
        value1 = Vector3d(1, 0, 0)
        value2 = Vector3d(2, 1, 0)
        np.testing.assert_almost_equal(value1.distbtw(value2), 2 ** 0.5)

    def test_xyz_1(self):
        value = Vector3d([1, 2, 3])
        self.assertEqual(value.x, 1)
        self.assertEqual(value.y, 2)
        self.assertEqual(value.z, 3)

    def test_xyz_2(self):
        value = Vector3d(1, 2, 3)
        self.assertEqual(value.x, 1)
        self.assertEqual(value.y, 2)
        self.assertEqual(value.z, 3)

    def test_sub_1(self):
        value1 = Vector3d(1, 2, 3)
        value2 = Vector3d(3, 2, 1)
        subed = value1 - value2
        answer = Vector3d(-2, 0, 2)
        self.assertEqual(subed, answer)
        self.assertEqual(subed.x, answer.x)
