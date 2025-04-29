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

cap = cv2.VideoCapture(0)

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
print(model)

lms = sorted([11, 13, 15, 23, 25, 27, 12, 14, 16, 24, 26, 28])

count = 0
attempt = False
holistic = mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)
while True:
#LOADING IMAGE
    ret,frame = cap.read()
    height , width,_ = frame.shape
    #print(height, width)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable=False

    result = holistic.process(rgb_frame)
    ########################
    num_coords = len(result.pose_landmarks.landmark)

    rgb_frame.flags.writeable=True
    rgb_frame= cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(rgb_frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

    mp_drawing.draw_landmarks(rgb_frame,result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

    mp_drawing.draw_landmarks(rgb_frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

    # Plot pose world landmarks.
    #mp_drawing.plot_landmarks(result.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
    try:

        pose = result.pose_landmarks.landmark
        pose = [pose[i-1] for i in lms]
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        row =  pose_row

        #make detection
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        #print(body_language_class, body_language_prob)

        #grab ear coords
        coords = tuple(np.multiply(np.array((result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)) , [640,480]).astype(int))

        cv2.rectangle(rgb_frame, (coords[0], coords[1]+5), (coords[0]+len(body_language_class)*20, coords[1]-30), (43, 41, 40), -1)

        cv2.putText(rgb_frame, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (3, 12, 138), 2, cv2.LINE_AA)

        # Get status box
        cv2.rectangle(rgb_frame, (0,0), (250, 60), (43, 41, 40), -1)
        # Display Probability
        cv2.putText(rgb_frame, 'PROB' , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 12, 138), 1, cv2.LINE_AA)
        cv2.putText(rgb_frame, str(round(body_language_prob[np.argmax(body_language_prob)],2)) , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (43,43,255), 2, cv2.LINE_AA)

        if round(body_language_prob[np.argmax(body_language_prob)]) < 0.6:
            cv2.putText(rgb_frame, 'Get into initial position' , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 12, 138), 1, cv2.LINE_AA)
        else :
            # Display Class
            cv2.putText(rgb_frame, 'CLASS' , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 12, 138), 1, cv2.LINE_AA)
            cv2.putText(rgb_frame, body_language_class.split(' ')[0] , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (43,43,255), 2, cv2.LINE_AA)

        # Display counter
        cv2.putText(rgb_frame, 'COUNTER: ' + str(count) , (90,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 12, 138), 1, cv2.LINE_AA)
        if body_language_class == "PUSHup":
            attempt = True
        elif body_language_class == "PUSHdown" and attempt:
                count += 1
                attempt = False
    except:
        pass

    if (cv2.waitKey(25) & 0xFF) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    cv2.imshow("MEDIAPIPE FRAME",rgb_frame)

cap.release()
cv2.destroyAllWindows()
