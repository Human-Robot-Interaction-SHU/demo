import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult

handSign = ""
handPosition: NormalizedLandmark = NormalizedLandmark(x=0.5, y=0.5)


def setResults(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures and result.gestures[0]:
        global handSign
        global handPosition
        handSign = result.gestures[0][0].category_name
        handPosition = result.hand_landmarks[0][0]


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

model_path = ('/Users/appsamuraiappsamurai/Documents/Social-Cue-Analysis-for-Human-Robot-Interaction'
              '/PoseDetectionModule/gesture_recognizer.task')

base_options = BaseOptions(model_asset_path=model_path)
VisionRunningMode = mp.tasks.vision.RunningMode
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                          result_callback=setResults)
sign_recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_number = 0
while cap.isOpened():
    # read frame
    frame_number += 1
    x, frame = cap.read()
    try:
        # resize the frame for portrait video
        # frame = cv2.resize(frame, (350, 600))
        # convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        sign_recognizer.recognize_async(mp_image, frame_number)

        pose_results = pose.process(frame_rgb)

        # draw skeleton on the frame
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        image_height, image_width, _ = frame.shape
        cx, cy = handPosition.x * image_width, handPosition.y * image_height

        frame = cv2.putText(img=frame, text=handSign, org=(int(cx), int(cy)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255, 0, 0), thickness=3)

        # display the frame
        cv2.imshow('Output', frame)
    except Exception as error:
        print(error)
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()