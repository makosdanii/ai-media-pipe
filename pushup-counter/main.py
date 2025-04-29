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

class_name = "STANCE"

cap = cv2.VideoCapture(0)

df = pd.read_csv("coord.csv")
#print(df.head())
#print(df[df["class"]=="sad"])

x = df.drop("class", axis=1) #features
y = df["class"] # target value

x_train , x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=1234)

pipelines = {
        "lr" : make_pipeline(StandardScaler(), LogisticRegression()),
        "rc": make_pipeline(StandardScaler(), RidgeClassifier()),
        "rf" : make_pipeline(StandardScaler(), RandomForestClassifier()),
        "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier()), }


fit_models = {}
for algo, pipeline in pipelines.items():
        model = pipeline.fit(x_train, y_train)
        fit_models[algo] = model

print(fit_models)
print(fit_models["rc"].predict(x_test))



for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test, yhat))

print(fit_models['rf'].predict(x_test))
print(y_test)

'''
with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)
'''
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
print(model)


with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while True:
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
                        #print(num_coords)

                        Landmarks = ["class"]
                        for val in range(1, num_coords+1):
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
                            '''
                            row.insert(0, class_name)
                            print(row)
                            #print(len(pose_row))
                            #print(len(face_row))


                            with open("coord.csv", mode="a", newline="") as f:
                                    csv_writer = csv.writer(f, delimiter=",", quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                                    csv_writer.writerow(row) '''

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

                            # Display Class
                            cv2.putText(rgb_frame, 'CLASS' , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 12, 138), 1, cv2.LINE_AA)

                            cv2.putText(rgb_frame, body_language_class.split(' ')[0] , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (43,43,255), 2, cv2.LINE_AA)

                            # Display Probability
                            cv2.putText(rgb_frame, 'PROB' , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 12, 138), 1, cv2.LINE_AA)
                            cv2.putText(rgb_frame, str(round(body_language_prob[np.argmax(body_language_prob)],2)) , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (43,43,255), 2, cv2.LINE_AA)



                        except:
                            pass

                        cv2.imshow("MEDIAPIPE FRAME",rgb_frame)

                        cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
