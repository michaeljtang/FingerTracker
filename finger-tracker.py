# finger tracking
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
import csv
import os
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
mp_hands = mp.solutions.hands

# INPUTS

# Define the path to the input video file here:
# video_file_name = './test_images/5.mp4'
video_file_name = './test_images/IMG_1286.MOV'

# Define the desired path to the output csv file containing finger coordinates here:
csv_output_name = './coords.csv'

###



# This changes the output csv file name to a name that isn't already in use
file_no=0
while os.path.isfile(csv_output_name[:-4]+"_"+str(file_no)+".csv"):
    file_no+=1
csv_output_name = csv_output_name[:-4]+"_"+str(file_no)+".csv"

cap = cv2.VideoCapture(video_file_name)
# Initiate holistic model
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        pose = results.multi_hand_landmarks
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                Thumb = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
                Index = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                Middle = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                Ring = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                Pinky = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_TIP]
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )
                T = list(
                    np.array([[Thumb.x, Thumb.y, Thumb.z] for landmark in pose]).flatten())
                I = list(
                    np.array([[Index.x, Index.y, Index.z] for landmark in pose]).flatten())
                M = list(
                    np.array([[Middle.x, Middle.y, Middle.z] for landmark in pose]).flatten())
                R = list(
                    np.array([[Ring.x, Ring.y, Ring.z] for landmark in pose]).flatten())
                P = list(
                    np.array([[Pinky.x, Pinky.y, Pinky.z] for landmark in pose]).flatten())

                    # Concate rows
                row = T+I+M+R+P
                class_name = 'Video1'
                    # Append class name
                row.insert(0, class_name)
                num_coords = len(pose)
                landmarks = ['Frame']
                for val in range(1, num_coords + 1):
                    landmarks += ['ThumbX{}'.format(val),'ThumbY{}'.format(val),'ThumbZ{}'.format(val),
                                  'IndexX{}'.format(val), 'IndexY{}'.format(val),'IndexZ{}'.format(val),
                                  'MiddleX{}'.format(val), 'MiddleY{}'.format(val),'MiddleZ{}'.format(val),
                                  'RingX{}'.format(val),'RingY{}'.format(val),'RingZ{}'.format(val),
                                  'PinkyX{}'.format(val),'PinkyY{}'.format(val),'PinkyZ{}'.format(val)]
                    # print(landmarks)
                """
                this is going to write the landmarks"['class', 'x1', 'y1', 'z1', 'v1', 'x2', 'y2', 'z2', 'v2', 'x3', 'y3', 'z3', 'v3', 'x4', 'y4', 'z4', 'v4', 'x5', 'y5', 'z5', 'v5', 'x6', 'y6', 'z6', 'v6', 'x7', 'y7"
                to the csv(excel) named "coords" in the same folder
                use this next line only to create new csv file.it will only create one line of data,
                 idealy: the step will be uncomment this line, 
                 commnent the next append csv line and then the file will be cleared.
                """

                #If the coordinates file doesn't already exist
                if not os.path.isfile(csv_output_name):
                    with open(csv_output_name, mode='w', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(landmarks)

                # # #Export to CSV
                with open(csv_output_name, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()