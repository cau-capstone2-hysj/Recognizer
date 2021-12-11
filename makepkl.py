# import pickle as pkl
# import dill
import pickle as pkl

import cloudpickle as cpkl

from components.mp_handler.mp_holistic import MpHolistic

mph = MpHolistic(image_directory=r"tests\resource\IU_0.jpg")
while 1:
    mpr = mph.process()
    if mpr.holistic_landmarks.hand["left"] or mpr.holistic_landmarks.hand["right"]:
        break
    print("retry..")

with open("tests/resource/IU_0_raw_result.pkl", "wb") as f:
    cpkl.dump(mpr.raw_results, f)

with open("tests/resource/IU_0_raw_result.pkl", "rb") as f:
    p = pkl.load(f)
    print(f"{p.pose_world_landmarks=}")

mph.imshow_result()
