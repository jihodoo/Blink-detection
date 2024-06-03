import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import time

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_eye_distance(eye):
    eye_width = distance.euclidean(eye[0], eye[3])
    actual_eye_width_cm = 3.0
    focal_length = 500
    distance_cm = (actual_eye_width_cm * focal_length) / eye_width
    return distance_cm

LEFT_EYE_START, LEFT_EYE_END = 42, 48
RIGHT_EYE_START, RIGHT_EYE_END = 36, 42

EAR_THRESHOLD =  0.22
EAR_CONSEC_FRAMES = 1
#채영님 0.2

blink_counter = 0
total_blinks = 0

# Time tracking variables for blinks per minute calculation
start_time = time.time()
current_minute_blinks = 0
minute_start_time = start_time
blink_rates = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    eyes_detected = False

    if len(rects) == 0:
        # 얼굴을 찾지 못한 경우, 눈만 감지
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        eye_centers = []
        for (x, y, w, h) in eyes:
            eye_center = (x + w // 2, y + h // 2)
            eye_radius = int(round((w + h) * 0.25))
            eye_centers.append((eye_center, eye_radius))
            eyes_detected = True

        # 두 개의 눈이 감지된 경우에만 눈 깜빡임 및 거리 측정을 수행
        if len(eye_centers) == 2:
            left_eye_center, left_eye_radius = eye_centers[0]
            
            right_eye_center, right_eye_radius = eye_centers[1]

            left_eye = np.array([
                [left_eye_center[0] - left_eye_radius, left_eye_center[1]],
                [left_eye_center[0], left_eye_center[1] - left_eye_radius],
                [left_eye_center[0] + left_eye_radius, left_eye_center[1]],
                [left_eye_center[0], left_eye_center[1] + left_eye_radius],
                [left_eye_center[0] - left_eye_radius // 2, left_eye_center[1] - left_eye_radius // 2],
                [left_eye_center[0] + left_eye_radius // 2, left_eye_center[1] + left_eye_radius // 2]
            ])

            right_eye = np.array([
                [right_eye_center[0] - right_eye_radius, right_eye_center[1]],
                [right_eye_center[0], right_eye_center[1] - right_eye_radius],
                [right_eye_center[0] + right_eye_radius, right_eye_center[1]],
                [right_eye_center[0], right_eye_center[1] + right_eye_radius],
                [right_eye_center[0] - right_eye_radius // 2, right_eye_center[1] - right_eye_radius // 2],
                [right_eye_center[0] + right_eye_radius // 2, right_eye_center[1] + right_eye_radius // 2]
            ])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    total_blinks += 1
                    current_minute_blinks += 1
                blink_counter = 0

            left_eye_distance = calculate_eye_distance(left_eye)
            right_eye_distance = calculate_eye_distance(right_eye)
            average_distance = (left_eye_distance + right_eye_distance) / 2.0

            cv2.putText(frame, f"Distance: {average_distance:.2f} cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            for (center, radius) in eye_centers:
                cv2.circle(frame, center, radius, (255, 0, 0), 2)

    else:
        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = shape[LEFT_EYE_START:LEFT_EYE_END]
            right_eye = shape[RIGHT_EYE_START:RIGHT_EYE_END]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    total_blinks += 1
                    current_minute_blinks += 1
                blink_counter = 0

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            left_eye_distance = calculate_eye_distance(left_eye)
            right_eye_distance = calculate_eye_distance(right_eye)
            average_distance = (left_eye_distance + right_eye_distance) / 2.0

            cv2.putText(frame, f"Distance: {average_distance:.2f} cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Check if one minute has passed
    current_time = time.time()
    if current_time - minute_start_time >= 60:
        blink_rates.append(current_minute_blinks)
        minute_start_time = current_time
        current_minute_blinks = 0

    # Calculate average blinks per minute
    if len(blink_rates) > 0:
        average_blinks_per_minute = sum(blink_rates) / len(blink_rates)
    else:
        average_blinks_per_minute = 0

    cv2.putText(frame, "Blinks: {}".format(total_blinks), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Blinks/min: {:.2f}".format(average_blinks_per_minute), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #신호등
    if average_blinks_per_minute <=7:
        color = (0,0,255) #red
        status = "Warning"
    elif 7< average_blinks_per_minute <= 15:
        color = (0,255,255) #yellow
        status = "Caution"
    else:
        color = (0,255,0) #gren
        status = "Good"
    
    cv2.rectangle(frame,(10,120),(50,160),color, -1)
    cv2.putText(frame,status,(60,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    cv2.imshow("Blink Detection and Distance Measurement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
