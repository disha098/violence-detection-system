import cv2
import numpy as np

IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

def extract_all_frames(video_path):

    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frame = frame / 255
        frames.append(frame)

    cap.release()

    print("Total frames extracted:", len(frames))
    return frames