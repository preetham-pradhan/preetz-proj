import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

cap = cv2.VideoCapture(0)

# Separate counters and stages
pushup_counter = 0
pushup_stage = None

squat_counter = 0
squat_stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (50, 50, 50), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        try:
            landmarks = results.pose_landmarks.landmark

            # PUSH-UP ANGLE (Elbow)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]

            pushup_angle = calculate_angle(shoulder, elbow, wrist)

            if pushup_angle > 160:
                pushup_stage = "up"
            if pushup_angle < 90 and pushup_stage == "up":
                pushup_stage = "down"
                pushup_counter += 1

            # SQUAT ANGLE (Knee)
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]

            squat_angle = calculate_angle(hip, knee, ankle)

            if squat_angle > 160:
                squat_stage = "up"
            if squat_angle < 100 and squat_stage == "up":
                squat_stage = "down"
                squat_counter += 1

            # ---- DISPLAY ----
            cv2.putText(frame, "Exercise Tracker", (width//2 - 200, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)

            # Push-ups (Top Left)
            cv2.putText(frame, f'Push-ups: {pushup_counter}', (30, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

            # Squats (Top Right)
            cv2.putText(frame, f'Squats: {squat_counter}', (width - 300, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 0), 3)

            # Calorie Burnt (Bottom Center)
            calories = (pushup_counter * 0.3) + (squat_counter * 0.32)
            cv2.putText(frame, f'Calories Burnt: {calories:.2f}', (width//2 - 250, height - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=6),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4)
            )

        except:
            pass

        cv2.imshow('Exercise Tracker', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
