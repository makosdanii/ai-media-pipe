import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle

#MP CONFIGS
mp_drawing = mp.solutions.drawing_utils
mp_holistic= mp.solutions.holistic

#DRWAING STYLES:
drawing_spec_LF = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color = (3, 252, 244))
drawing_spec_CF = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color = (251, 255, 122))

#Change index depending on the camera being used (0, 1 or 2)
cap = cv2.VideoCapture(0)

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
print(model)

#Filtered Mediapipe body landmarks (body points)
lms = sorted([11, 13, 15, 23, 25, 27, 12, 14, 16, 24, 26, 28])
count = 0
attempt = False
holistic = mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

while True:
    #Loading image from camera
    ret,frame = cap.read()
    if not ret:
        exit(1)
    height , width,_ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable=False

    result = holistic.process(rgb_frame)

    rgb_frame.flags.writeable=True
    rgb_frame= cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    #Case: no landmarks detected
    if not result.pose_landmarks:
        cv2.rectangle(rgb_frame, (0,0), (250, 60), (43, 41, 40), -1)
        cv2.putText(rgb_frame, 'No person detected' , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("MEDIAPIPE FRAME",rgb_frame)
        continue

    #Display landmarks using Mediapipe
    num_coords = len(result.pose_landmarks.landmark)
    mp_drawing.draw_landmarks(rgb_frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)
    mp_drawing.draw_landmarks(rgb_frame,result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)
    mp_drawing.draw_landmarks(rgb_frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec_CF,connection_drawing_spec=drawing_spec_LF)

    #Collecting landmark data for prediction
    pose = result.pose_landmarks.landmark
    pose = [pose[i-1] for i in lms]
    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    row = pose_row

    #Prediction
    X = pd.DataFrame([row])
    body_language_class = model.predict(X)[0]
    body_language_prob = model.predict_proba(X)[0]

    # Get status box
    cv2.rectangle(rgb_frame, (0,0), (250, 60), (43, 41, 40), -1)
    # Display Probability
    cv2.putText(rgb_frame, 'PROB' , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(rgb_frame, str(round(body_language_prob[np.argmax(body_language_prob)],2)) , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    #Check accuracy
    if round(body_language_prob[np.argmax(body_language_prob)], 2) < 0.65:
        cv2.putText(rgb_frame, 'Get into initial position' , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    else :
        # Display Class
        cv2.putText(rgb_frame, 'CLASS' , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(rgb_frame, body_language_class.split(' ')[0] , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    #Display counter
    cv2.putText(rgb_frame, 'COUNTER: ' + str(count) , (0,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    if body_language_class == "PUSHup":
        attempt = True
    elif body_language_class == "PUSHdown" and attempt:
            count += 1
            attempt = False
    
    #Press 'Q' key to exit window
    if (cv2.waitKey(25) & 0xFF) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    cv2.imshow("MEDIAPIPE FRAME",rgb_frame)

cap.release()
cv2.destroyAllWindows()
