import os
import cv2
import random
import shutil
from  pathlib import Path

#Folder structure
# data
#  train
#    images
#    labels
#  valid
#    images
#    labels
#  test
#    images
#    labels

# Extract frames from video and save some of them as validation data and some as test data 
def extract_frames_for_dataset(video_path, data_path, valid_pct = 0.2, test_pct = 0.0, max_frames = None, skip_frames = 1):
    # Create folders
    os.makedirs(data_path + "/train/images", exist_ok=True)
    os.makedirs(data_path + "/train/labels", exist_ok=True)
    os.makedirs(data_path + "/valid/images", exist_ok=True)
    os.makedirs(data_path + "/valid/labels", exist_ok=True)
    os.makedirs(data_path + "/test/images", exist_ok=True)
    os.makedirs(data_path + "/test/labels", exist_ok=True)
    
    # Open the video
    vidcap = cv2.VideoCapture(video_path)
    #rotate image cw 
    count = 0

    is_valid_frame_added = False
    is_test_frame_added = False

    while True:
        success,image = vidcap.read()
        if not success:
            break
        
        if count % skip_frames != 0:
            count += 1
            continue
            
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        if random.random() < valid_pct or not is_valid_frame_added:
            folder = "valid"
            is_valid_frame_added = True
        elif random.random() < test_pct or not is_test_frame_added:
            folder = "test"
            is_test_frame_added = True
        else:
            folder = "train"
        cv2.imwrite(f"{data_path}/{folder}/images/frame{count}.jpg", image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    print(f"Extracted {count} frames from {video_path} to {data_path}/train/images, {data_path}/valid/images, {data_path}/test/images")

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    data_path = input("Enter the path to the data folder: ")    
    extract_frames_for_dataset(video_path, data_path, valid_pct = 0.2, test_pct = 0.0, max_frames = 32)
    
