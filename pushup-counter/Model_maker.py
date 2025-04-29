import mediapipe as mp
import cv2
import csv
import os
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score
import pickle


#MP CONFIGS
mp_drawing = mp.solutions.drawing_utils
mp_holistic= mp.solutions.holistic

#DRWAING STYLES:
drawing_spec_LF = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color = (3, 252, 244))
drawing_spec_CF = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color = (251, 255, 122))

class_name = "PUSHup" #CHANGE THIS CLASS NAME AND RESHOOT
cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    lms = sorted([11, 13, 15, 23, 25, 27, 12, 14, 16, 24, 26, 28])
    Landmarks = ["class"]
    for val in lms:
        Landmarks += ["x{}".format(val), "y{}".format(val), "z{}".format(val), "v{}".format(val)]
    #with open("coord.csv", mode="w", newline="") as f:
        #csv_writer = csv.writer(f, delimiter=",", quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        #csv_writer.writerow(Landmarks)
                        
    while True:
        ret,frame = cap.read()
        height , width,_ = frame.shape
        #print(height, width)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable=False

        # Make Detections
        result = holistic.process(rgb_frame)
        # print(results.face_landmarks)

        # Recolor image back to BGR for rendering
        rgb_frame.flags.writeable = True
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(rgb_frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

        mp_drawing.draw_landmarks(rgb_frame,result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

        mp_drawing.draw_landmarks(rgb_frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)
        pose = result.pose_landmarks.landmark
        # Export coordinates
        pose = [pose[i-1] for i in lms]
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        # Concate rows
        row = pose_row

        # Append class name
        row.insert(0, class_name)
        try:
            # Extract Pose landmarks
            if not os.path.exists('coord.csv'):
                print("Fuck")
            # Export to CSV
            with open('coord.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

        except:
            import traceback
            print(traceback.format_exc())

        cv2.imshow('Model Making Window', rgb_frame)
        cv2.waitKey(1)



cap.release()
cv2.destroyAllWindows()
