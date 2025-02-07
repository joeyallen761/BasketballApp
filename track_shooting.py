exit()#don't want to run this even on accident
import deeplabcut
import tensorflow as tf

import os
print(os.getcwd())

video = "videos/freethrows_38sec.mp4"
projectName = "Project4BasketballHumanProject"
experimenter = "dev1"
config_path=projectName+'-'+experimenter+'-2025-02-06/config.yaml'

# #OR, can I try a different yaml config path? ie the train cofig?, when analyze_videos is happening.
# #Ill try that first
# test_config_path = "Project4BasketballHumanProject-dev1-2025-02-06/dlc-models/iteration-0/Project4BasketballHumanProjectFeb6-trainset100shuffle1/test/pose_cfg.yaml"
# train_config_path = "Project4BasketballHumanProject-dev1-2025-02-06/dlc-models/iteration-0/Project4BasketballHumanProjectFeb6-trainset100shuffle1/train/pose_cfg.yaml"
# #new error but nothing working

path = deeplabcut.create_pretrained_project(
    project = projectName,
    experimenter = experimenter,
    videos=[video],
    copy_videos=True,
    model='full_human',
    videotype="mp4",
    trainFraction=1
)

# print(path)

# deeplabcut.label_frames(config_path)

# deeplabcut.extract_frames(config_path, mode='automatic')
# deeplabcut.check_dataset(config_path)

# deeplabcut.analyze_videos(
#     config_path,
#     ["Project4BasketballHumanProject-dev1-2025-02-06/videos/freethrows_38sec.mp4"],
#     videotype='.mp4',
#     save_as_csv=True
# )

