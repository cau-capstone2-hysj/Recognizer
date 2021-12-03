"""
create a chunk of orgainzed landmarks from raw landmarks
"""
from pprint import pformat
from typing import Dict

# from ..mp_handler.mediapipe_result import MediaPipeResult
from .consts import HAND_LANDMARK_ID_TO_NAME, POSE_LANDMARK_ID_TO_NAME
from .mylandmark import MyLandmark


class HolisticLandmarks:
    """
    process MediapipeResult to organized landmarks
    """

    def __init__(self, mediapipe_result) -> None:
        (
            __pose_worldlandmarks,
            __lefthand_landmarks,
            __righthand_landmarks,
        ) = HolisticLandmarks.__split_landmarks(mediapipe_result)
        self.__landmarks: dict = {
            "pose": {"left": {}, "right": {}},
            "hand": {"left": {}, "right": {}},
        }

        for i, pwlm in enumerate(__pose_worldlandmarks):
            landmark = MyLandmark(pwlm.x, pwlm.y, pwlm.z, pwlm.visibility)
            save_to = self.__landmarks["pose"]
            name = POSE_LANDMARK_ID_TO_NAME[i]
            if name.startswith("left"):
                save_to["left"][name[5:]] = landmark
            elif name.startswith("right"):
                save_to["right"][name[6:]] = landmark
            else:
                pass

        for i, hlm in enumerate(__lefthand_landmarks):
            landmark = MyLandmark(hlm.x, hlm.y)
            save_to = self.__landmarks["hand"]["left"]
            save_to[HAND_LANDMARK_ID_TO_NAME[i]] = landmark

        for i, hlm in enumerate(__righthand_landmarks):
            landmark = MyLandmark(hlm.x, hlm.y)
            save_to = self.__landmarks["hand"]["right"]
            save_to[HAND_LANDMARK_ID_TO_NAME[i]] = landmark

    @staticmethod
    def __split_landmarks(mp_results) -> tuple:
        """
        split results to landmarks and make values None-safe
        """
        pose_worldlandmarks, lefthand_landmarks, righthand_landmarks = [], [], []
        if mp_results.pose_world_landmarks:
            pose_worldlandmarks = mp_results.pose_world_landmarks.landmark
        if mp_results.left_hand_landmarks:
            lefthand_landmarks = mp_results.left_hand_landmarks.landmark
        if mp_results.right_hand_landmarks:
            righthand_landmarks = mp_results.right_hand_landmarks.landmark
        return pose_worldlandmarks, lefthand_landmarks, righthand_landmarks

    @property
    def whole(self) -> Dict[str, Dict[str, Dict[str, MyLandmark]]]:
        return self.__landmarks

    @property
    def pose(self) -> Dict[str, Dict[str, MyLandmark]]:
        return self.__landmarks["pose"]

    @property
    def hand(self) -> Dict[str, Dict[str, MyLandmark]]:
        return self.__landmarks["hand"]

    def __str__(self) -> str:
        return pformat(self.__landmarks)

    def __repr__(self) -> str:
        return self.__str__()
