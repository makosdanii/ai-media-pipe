import cv2
from mediapipe import solutions
import math

# based on the reference/pose_tracking_full_body_landmarks
left_lms = [11, 13, 15, 23, 25, 27]
right_lms = [12, 14, 16, 24, 26, 28]
lm_list = list()

mpDraw = solutions.drawing_utils
pose = solutions.pose.Pose(static_image_mode=False,
                                smooth_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

def findPose(img, pose=pose, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        if draw:
            mpDraw.draw_landmarks(img, results.pose_landmarks, solutions.pose.POSE_CONNECTIONS)
        h, w, c = img.shape
        return list(map(lambda lm: [lm[0], int(lm[1].x * w), int(lm[1].y * h)], 
                   enumerate(results.pose_landmarks.landmark)))
    else: return []

def findAngle(img, landmarks, lm_list=lm_list, draw=True):
    points = []
    if len(landmarks) != 3:
        print("Insufficient passed landmarks")
        return 0
    else:
        # filter for those mediapipe land points with ids (lm[0]) which are in the landmarks list
        points = list(filter(lambda lm: lm[0] in landmarks, lm_list))

    if len(points) != 3:
        print("Insufficient detected landmarks")
        print(lm_list)
        return 0

    # Get the landmark
    x1, y1 = points[0][1:]
    x2, y2 = points[1][1:]
    x3, y3 = points[2][1:]

    # Calculate the angle
    angle = math.degrees(
        math.fabs(math.atan2(math.fabs(y3 - y2), math.fabs(x3 - x2))) + 
        math.fabs(math.atan2(math.fabs(y1 - y2), math.fabs(x1 - x2))))
    # some time this angle comes zero, so below conditon we added
    # if angle < 0:
    #     angle += 360
    # if angle > 180: 
    #     angle -= 180

    # Draw
    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, (0, 0, 255), 1)
        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), 1)
        cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 15, (0, 0, 255), 1)
        cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    return angle

cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

pushups = 0
attempt = False
while True:
    success, img = cap.read()
    if not success:
        print("Failed to get the image")
        exit(1)

    lm_list.clear()
    lm_list .extend(findPose(img))

    angle_left_arm = findAngle(img, left_lms[0:3])
    angle_left_leg = findAngle(img, left_lms[3:6])
    angle_right_arm = findAngle(img, right_lms[0:3])    
    angle_right_leg = findAngle(img, right_lms[3:6])

    if angle_left_arm < 100 or angle_right_arm < 100:
        if attempt:
            pushups += 1
            attempt = False
    elif angle_left_arm > 160 or angle_right_arm > 160:
        attempt = True

    cv2.putText(img, "Push-ups: " + str(pushups), (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    5, (0, 0, 255), 2)
    cv2.imshow("Image", img)
    
    if (cv2.waitKey(25) & 0xFF) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
