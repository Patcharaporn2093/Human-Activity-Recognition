import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
 
# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
 
 
# Function to detect if the person is clapping
def is_clapping(pose_landmarks, threshold=0.05):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    return (
        abs(left_wrist.x - right_wrist.x) < threshold
        and abs(left_wrist.y - right_wrist.y) < threshold
    )
 
 
# Function to detect if either hand is raised
def is_raising_hand(pose_landmarks, hand="right"):
    wrist = pose_landmarks.landmark[
        (
            mp_pose.PoseLandmark.RIGHT_WRIST
            if hand == "right"
            else mp_pose.PoseLandmark.LEFT_WRIST
        )
    ].y
    shoulder = pose_landmarks.landmark[
        (
            mp_pose.PoseLandmark.RIGHT_SHOULDER
            if hand == "right"
            else mp_pose.PoseLandmark.LEFT_SHOULDER
        )
    ].y
    return wrist < shoulder
 
 
# # Function to detect if the person is standing
# def is_standing(pose_landmarks, threshold=0.05):
#     shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
#     hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
#     knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
#     return abs(shoulder - hip) < threshold and abs(hip - knee) < threshold
 
 
# Global variable to track the previous foot index position for walking detection
prev_foot_index_y = None
 
 
# Function to detect walking based on the change in distance between feet landmarks
def is_walking(pose_landmarks, threshold=0.0008):
    global prev_foot_index_y
    left_foot_index_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
    right_foot_index_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
 
    current_position = (left_foot_index_y + right_foot_index_y) / 2
    if prev_foot_index_y is not None:
        movement = abs(current_position - prev_foot_index_y)
        prev_foot_index_y = current_position
        return movement > threshold
    else:
        prev_foot_index_y = current_position
        return False


def detect_activity(pose_landmarks):
    if is_walking(pose_landmarks):
        return "Walking"
    elif is_clapping(pose_landmarks):
        return "Clapping"
    elif is_raising_hand(pose_landmarks, "right"):
        return "Raising Right Hand"
    elif is_raising_hand(pose_landmarks, "left"):
        return "Raising Left Hand"
    return "Standing"
 
 
if __name__ == "__main__":
    video = "video_test/video_test.mp4"
    cap = cv2.VideoCapture(video)
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
 
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
 
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            activity = detect_activity(results.pose_landmarks)
            cv2.putText(
                frame,
                activity,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 0),
                3,
                cv2.LINE_AA,
            )

        frame = cv2.resize(frame, (1000, 800))  # Set desired width and height
 
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()