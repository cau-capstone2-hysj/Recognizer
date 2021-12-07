"""
create a chunk of orgainzed landmarks from raw landmarks
"""
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, NamedTuple

# from ..mp_handler.mediapipe_result import MediaPipeResult
from mediapipe.framework.formats.landmark_pb2 import Landmark

from ..custom_typings import TypeRealNumber
from .consts import HAND_LANDMARK_ID_TO_NAME, POSE_LANDMARK_ID_TO_NAME
from .mylandmark import MyLandmark


def _parse_raw_mp_results(mp_results) -> tuple:
    """
    parse results to landmarks and make values None-safe
    """
    pose_worldlandmarks, lefthand_landmarks, righthand_landmarks = [], [], []
    if mp_results.pose_world_landmarks:
        pose_worldlandmarks = mp_results.pose_world_landmarks.landmark
    if mp_results.left_hand_landmarks:
        lefthand_landmarks = mp_results.left_hand_landmarks.landmark
    if mp_results.right_hand_landmarks:
        righthand_landmarks = mp_results.right_hand_landmarks.landmark
    return pose_worldlandmarks, lefthand_landmarks, righthand_landmarks


@dataclass
class LandmarkContainer:
    """
    simple container for landmarks
    (to replace dictionary-structed landmarks in Future)
    """

    left: dict = field(default_factory=dict)
    right: dict = field(default_factory=dict)


class HolisticLandmarks:
    """
    process MediapipeResult to organized landmarks
    """

    def __init__(self, raw_mediapipe_result: NamedTuple) -> None:
        (
            __pose_worldlandmarks,
            __lefthand_landmarks,
            __righthand_landmarks,
        ) = _parse_raw_mp_results(raw_mediapipe_result)

        self.__landmarks: dict = {
            "pose": {"left": {}, "right": {}},
            "hand": {"left": {}, "right": {}},
        }
        self.__lm_pose = LandmarkContainer()
        self.__lm_hand = LandmarkContainer()

        self.__process_pose_landmarks(__pose_worldlandmarks)
        self.__process_hand_landmarks(__lefthand_landmarks, __righthand_landmarks)

    def __process_pose_landmarks(self, pose_worldlandmarks) -> None:
        pwlm: Landmark
        for i, pwlm in enumerate(pose_worldlandmarks):
            x: TypeRealNumber
            y: TypeRealNumber
            z: TypeRealNumber
            vis: TypeRealNumber
            x, y, z, vis = pwlm.x, pwlm.y, pwlm.z, pwlm.visibility

            landmark = MyLandmark((x, y, z), vis)
            save_to = self.__landmarks["pose"]
            name = POSE_LANDMARK_ID_TO_NAME[i]
            if name.startswith("left"):
                save_to["left"][name[5:]] = landmark
                self.__lm_pose.left[name[5:]] = landmark
            elif name.startswith("right"):
                save_to["right"][name[6:]] = landmark
                self.__lm_pose.right[name[6:]] = landmark
            else:
                pass

    def __process_hand_landmarks(self, lefthand_landmarks, righthand_landmarks) -> None:
        for i, hlm in enumerate(lefthand_landmarks):
            landmark = MyLandmark((hlm.x, hlm.y))
            save_to = self.__landmarks["hand"]["left"]
            save_to[HAND_LANDMARK_ID_TO_NAME[i]] = landmark
            self.__lm_hand.left[HAND_LANDMARK_ID_TO_NAME[i]] = landmark

        for i, hlm in enumerate(righthand_landmarks):
            landmark = MyLandmark((hlm.x, hlm.y))
            save_to = self.__landmarks["hand"]["right"]
            save_to[HAND_LANDMARK_ID_TO_NAME[i]] = landmark
            self.__lm_hand.right[HAND_LANDMARK_ID_TO_NAME[i]] = landmark

    @property
    def whole(self) -> Dict[str, Dict[str, Dict[str, MyLandmark]]]:
        return self.__landmarks

    # TODO: change dictionarys to LandmarkContainers
    @property
    def pose(self) -> Dict[str, Dict[str, MyLandmark]]:
        return self.__landmarks["pose"]

    @property
    def hand(self) -> Dict[str, Dict[str, MyLandmark]]:
        return self.__landmarks["hand"]

    def __str__(self) -> str:
        return pformat(self.__landmarks)

    def __repr__(self) -> str:
        return self.__landmarks.__repr__()
