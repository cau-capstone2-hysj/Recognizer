import logging
from math import atan2, pi
from time import sleep
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from components.landmark.holistic_landmarks import HolisticLandmarks

from .landmark.mylandmark import MyLandmark
from .landmark.vector3d import Vector3d
from .mp_handler.mp_holistic import MpHolistic

__POSE_FILTER_LIST = ["shoulder", "elbow", "wrist", "pinky", "index"]
__HAND_FILTER_LIST = ["INDEX_FINGER_TIP", "THUMB_TIP"]


def _pose_filter(pose: Dict[str, MyLandmark]) -> Dict[str, MyLandmark]:
    return {name: pose[name] for name in __POSE_FILTER_LIST if name in pose}


def _hand_filter(hand: Dict[str, MyLandmark]) -> Dict[str, MyLandmark]:
    return {name: hand[name] for name in __HAND_FILTER_LIST if name in hand}


class RecognizedArm(NamedTuple):
    image: np.ndarray
    theta: List[int]


class MyArm:
    """
    Returns RecognizedArm from MpHolistic
    """

    def __init__(self, is_rightarm=True, vis_threshold=0.80, image_dir=None) -> None:
        self.__armside = "right" if is_rightarm else "left"
        self.__vis_threshold = vis_threshold
        self.__mp = MpHolistic(
            image_dir, min_detection_confidence=0.5, min_tracking_confidence=0.9
        )

    def __get_mpr(self):
        return self.__mp.process()

    def __is_pose_data_available(self, landmark: HolisticLandmarks) -> bool:
        return bool(landmark.pose[self.__armside])

    def __is_hand_data_available(self, landmark: HolisticLandmarks) -> bool:
        return bool(landmark.hand[self.__armside])

    def __get_data(self):
        """
        filter and check if data is available and return data
        """
        while 1:
            mpr = self.__get_mpr()
            img, landmarks = mpr.marked_image, mpr.holistic_landmarks

            if not self.__is_pose_data_available(landmarks):
                logging.warning("No pose detected")
                # continue
            if not self.__is_hand_data_available(landmarks):
                logging.warning("No hand detected")
                # continue

            pose_filtered = _pose_filter(landmarks.pose[self.__armside])
            hand_filtered = _hand_filter(landmarks.hand[self.__armside])

            is_visible = all(
                p.vis >= self.__vis_threshold for p in pose_filtered.values()
            )

            if not is_visible:
                warning_msg = "Not all landmarks are visible: \n"
                unavailable_poses = [
                    f"{pose} is not visible, {vis}"
                    for pose, vis in pose_filtered.items()
                    if vis.vis < self.__vis_threshold
                ]
                warning_msg += "\n".join(unavailable_poses)
                logging.warning(warning_msg)
                # continue

            return img, pose_filtered, hand_filtered

    def process(self):
        img, pose, hand = self.__get_data()
        if pose and hand:
            pose, hand = map(_landmarks_to_vectors, (pose, hand))

            origin = pose["elbow"]
            pose = _translate_vectors(pose, origin)
            logging.info(
                f"elbow: {pose['elbow']}, wrist: {pose['wrist']}, elbow-wrist: {round(Vector3d(pose['elbow'].x, pose['elbow'].y, 0).distbtw(Vector3d(pose['wrist'].x, pose['wrist'].y, 0)),2)}"
            )

            handtip = Vector3d((pose["index"] + pose["pinky"]) / 2)

            # elbowToWrist_projected = Vector3d(
            #     pose["wrist"] - np.array([0, 0, pose["wrist"].z])
            # )
            elbowToWrist_projected = Vector3d(pose["wrist"].x, 0, pose["wrist"].z)

            wristToHandTip = Vector3d(handtip - pose["wrist"])
            dist_btw_thumb_index = hand["THUMB_TIP"].distbtw(hand["INDEX_FINGER_TIP"])

            # theta0 = Vector3d([0, 0, 1]).anglebtw(elbowToWrist_projected)
            theta0 = atan2(pose["wrist"].z, pose["wrist"].x) * 180 / pi
            # theta0 = _distance_to_angle(-0.3, 0.3, 170, 10, pose["wrist"].z)
            # theta1 = pose["wrist"].anglebtw(Vector3d(0, -1, 0))
            theta1 = _distance_to_angle(0, -0.2, 0, 90, pose["wrist"].y)
            theta2 = pose["wrist"].anglebtw(wristToHandTip)
            # theta3 = _distance2angle3(dist_btw_thumb_index)
            theta3 = _distance_to_angle(0, 0.26, 60, 0, dist_btw_thumb_index)
            thetas = [theta0, theta1, theta2, theta3]
            return RecognizedArm(img, list(map(int, thetas)))
        else:
            return RecognizedArm(img, [-1, -1, -1, -1])


def _translate_vectors(
    vectors: Dict[str, Vector3d], origin: Vector3d
) -> Dict[str, Vector3d]:
    return {k: v - origin for k, v in vectors.items()}


def _landmarks_to_vectors(landmarks: Dict[str, MyLandmark]) -> Dict[str, Vector3d]:
    return {k: Vector3d(v.coord) for k, v in landmarks.items()}


def _distance_to_angle(
    min_dist: float, max_dist: float, min_angle: float, max_angle: float, dist: float
) -> float:
    value = min_angle + (dist - min_dist) / (max_dist - min_dist) * (
        max_angle - min_angle
    )
    return min(max(value, 0), 180)
