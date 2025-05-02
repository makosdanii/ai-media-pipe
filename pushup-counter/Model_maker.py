import mediapipe as mp
import cv2
import csv
import numpy as np
import argparse
import traceback

#MediaPipe configs
mp_drawing = mp.solutions.drawing_utils
mp_holistic= mp.solutions.holistic

#Drawing style for MediaPipe
drawing_spec_LF = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color = (3, 252, 244))
drawing_spec_CF = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color = (251, 255, 122))

parser = argparse.ArgumentParser()
#Argument for the class type you are recording
parser.add_argument('--position', type=str, required=True)
#Argument for erasing the csv file
parser.add_argument('--erase', type=bool, required=False)
class_name = parser.parse_args().position
erase = parser.parse_args().erase

#Change index depending on the camera being used (0, 1 or 2)
cap = cv2.VideoCapture(1)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    lms = sorted([11, 13, 15, 23, 25, 27, 12, 14, 16, 24, 26, 28])
    Landmarks = ["class"]
    for val in lms:
        Landmarks += ["x{}".format(val), "y{}".format(val), "z{}".format(val), "v{}".format(val)]
    if erase:
        with open("coord.csv", mode="w", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",", quotechar = '"', quoting = csv.QUOTE_MINIMAL)
            csv_writer.writerow(Landmarks)
                        
    while True:
        ret,frame = cap.read()
        height , width,_ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable=False

        # Make Detections
        result = holistic.process(rgb_frame)

        # Recolor image back to BGR for rendering
        rgb_frame.flags.writeable = True
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        #Display landmarks using Mediapipe
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
            # Export to CSV
            with open('coord.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
        except:
            print(traceback.format_exc())

        cv2.imshow('Model Making Window', rgb_frame)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
