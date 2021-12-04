import logging
from time import sleep
from typing import Dict, List, NamedTuple

import numpy as np

from ..landmark.mylandmark import MyLandmark
from ..landmark.vector3d import Vector3d
from ..mp_handler.mp_holistic import MpHolistic


class RecognizedArm(NamedTuple):
    image: np.ndarray
    pose: Dict[str, MyLandmark]
    hand: Dict[str, MyLandmark]
    d: List[float]
    theta: List[float]


class MyArm:
    def __init__(self, is_rightarm=True, vis_threshold=0.80, image_dir=None) -> None:
        self.__armside = "right" if is_rightarm else "left"
        self.__vis_threshold = vis_threshold
        self.__mp = MpHolistic(image_dir)

    def __get_mpr(self):
        return self.__mp.process()

    def process(self):

        while 1:
            mpr = self.__get_mpr()
            img, landmarks = mpr.marked_image, mpr.holistic_landmarks
            if not landmarks.pose[self.__armside]:
                logging.warning("No pose detected")
                sleep(0.5)
                continue
            if not landmarks.hand[self.__armside]:
                logging.warning("No hand detected")
                sleep(0.5)
                continue
            pose = {
                i: landmarks.pose[self.__armside][i]
                for i in ["shoulder", "elbow", "wrist", "pinky", "index"]
            }
            hand = {
                i: landmarks.hand[self.__armside][i]
                for i in ["INDEX_FINGER_TIP", "THUMB_TIP"]
            }

            is_visible = all(p.vis >= self.__vis_threshold for p in pose.values())
            if not is_visible:
                logging.warning("Not all landmarks are visible")
                warning = ""
                for p in pose:
                    if pose[p].vis < self.__vis_threshold:
                        warning += f"{p} is not visible, {pose[p].vis} \n"
                logging.warning(warning)
                sleep(0.5)
                continue

            # print(f'{pose["wrist"]=}')
            origin = pose["elbow"].np
            for p in pose:
                v = pose[p]
                pose[p] = Vector3d((v.np - origin))
            handTip = Vector3d((pose["index"].np + pose["pinky"].np) / 2)

            elbowToWrist_projected = Vector3d(
                pose["wrist"].np - np.array([0, 0, pose["wrist"].z])
            )

            wristToHandTip = Vector3d(handTip.np - pose["wrist"].np)

            theta0 = Vector3d([1, 0, 0]).anglebtw(elbowToWrist_projected)
            theta1 = pose["wrist"].anglebtw(Vector3d(0, -1, 0))
            theta2 = pose["wrist"].anglebtw(wristToHandTip)
            d = hand["THUMB_TIP"].distbtw(hand["INDEX_FINGER_TIP"])
            return RecognizedArm(img, pose, hand, [d], [theta0, theta1, theta2])
