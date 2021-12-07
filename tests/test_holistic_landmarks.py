import pickle as pkl
import unittest

from components.landmark.consts import (
    HAND_LANDMARK_NAME_TO_ID,
    POSE_LANDMARK_NAME_TO_ID,
)
from components.landmark.holistic_landmarks import HolisticLandmarks


class MpHolisticTest(unittest.TestCase):
    def setUp(self) -> None:
        with open("tests/resource/IU_1_raw_result.pkl", "rb") as f:
            self.raw_result = pkl.load(f)
        self.hlm = HolisticLandmarks(self.raw_result)

    def test_pose_left_eye(self):
        idx = POSE_LANDMARK_NAME_TO_ID["left_eye"]
        value1 = self.hlm.pose["left"]["eye"]
        value2 = self.raw_result.pose_world_landmarks.landmark[idx]
        self.assertAlmostEqual(value1.x, value2.x)
        self.assertAlmostEqual(value1.y, value2.y)
        self.assertAlmostEqual(value1.z, value2.z)
        self.assertAlmostEqual(value1.vis, value2.visibility)

    def test_pose_right_index(self):
        idx = POSE_LANDMARK_NAME_TO_ID["right_index"]
        value1 = self.hlm.pose["right"]["index"]
        value2 = self.raw_result.pose_world_landmarks.landmark[idx]
        self.assertAlmostEqual(value1.x, value2.x)
        self.assertAlmostEqual(value1.y, value2.y)
        self.assertAlmostEqual(value1.z, value2.z)
        self.assertAlmostEqual(value1.vis, value2.visibility)

    def test_lefthand_THUMB_TIP(self):
        idx = HAND_LANDMARK_NAME_TO_ID["THUMB_TIP"]
        value1 = self.hlm.hand["left"]["THUMB_TIP"]
        value2 = self.raw_result.left_hand_landmarks.landmark[idx]
        self.assertAlmostEqual(value1.x, value2.x)
        self.assertAlmostEqual(value1.y, value2.y)

    def test_righthand_THUMB_TIP(self):
        idx = HAND_LANDMARK_NAME_TO_ID["PINKY_DIP"]
        value1 = self.hlm.hand["right"]["PINKY_DIP"]
        value2 = self.raw_result.right_hand_landmarks.landmark[idx]
        self.assertAlmostEqual(value1.x, value2.x)
        self.assertAlmostEqual(value1.y, value2.y)
