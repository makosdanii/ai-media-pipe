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

class_name = "Curl"

#MP CONFIGS
mp_drawing = mp.solutions.drawing_utils
mp_holistic= mp.solutions.holistic

#DRWAING STYLES:
drawing_spec_LF = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color = (3, 252, 244))
drawing_spec_CF = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color = (251, 255, 122))

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        x=0
        while x<1:
                    #LOADING IMAGE
                        ret,frame = cap.read()
                        height , width,_ = frame.shape
                        #print(height, width)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_frame.flags.writeable=False

                        #HAND_LANDMARKS
                        result = holistic.process(rgb_frame)
                        ########################
                        num_coords = len(result.pose_landmarks.landmark)

                        Landmarks = ["class"]
                        lms = [11, 13, 15, 23, 25, 27, 12, 14, 16, 24, 26, 28]
                        for val in lms:
                                Landmarks += ["x{}".format(val), "y{}".format(val), "z{}".format(val), "v{}".format(val)]
                        #print(Landmarks)



                        rgb_frame.flags.writeable=True
                        rgb_frame= cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                        #DRAWING POSE, HAND[LEFT AND RIGHT], FACE LANDMARKS

                       # if result.pose_landmarks:
                                #print(f'Nose coordinates: ('f'{result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * width}, 'f'{result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * height})')




                        mp_drawing.draw_landmarks(rgb_frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

                        mp_drawing.draw_landmarks(rgb_frame,result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

                        mp_drawing.draw_landmarks(rgb_frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

                        # Plot pose world landmarks.
                        #mp_drawing.plot_landmarks(result.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

                        try:

                            pose = result.pose_landmarks.landmark
                            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())



                            row =  pose_row

                            row.insert(0, class_name)
                            print(row)
                            #print(len(pose_row))
                            #print(len(face_row))


                            with open("coord.csv", mode="w", newline="") as f:
                                    csv_writer = csv.writer(f, delimiter=",", quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                                    csv_writer.writerow(Landmarks)
                        except:
                                pass

                        cv2.imshow('Model Making Window', rgb_frame)
                        cv2.waitKey(1)
                        x+=1



cap.release()
cv2.destroyAllWindows()
