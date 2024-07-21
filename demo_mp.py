import concurrent.futures

from modules.models import EyeTrackingForEveryone
from modules.AttentionModule import AttentionModule
import torch
import cv2 as cv
from screeninfo import get_monitors
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

# Pose detection
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult
from mediapipe.tasks.python import BaseOptions, vision
import mediapipe as mp

# Facial recognition
from deepface import DeepFace

# Get monitor size to set the size of the output image later
try:
    monitor = get_monitors()
    screen_width = monitor[0].width
    screen_height = monitor[0].height
except:
    print("No monitors found")
    screen_width=1920
    screen_height=1080

# Attention modules
model = EyeTrackingForEveryone()
model.load_state_dict(torch.load("./weights/attention_weights"))
attn = AttentionModule(model.to('cuda'))

# Pose detection

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

model_path = ('./weights/gesture_recognizer.task')

base_options = BaseOptions(model_asset_path=model_path)
VisionRunningMode = mp.tasks.vision.RunningMode
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=mp.tasks.vision.FaceDetectorOptions.running_mode)

sign_recognizer = vision.GestureRecognizer.create_from_options(options)
frame_number = 0

# end pose detection


# facial expression

# initialize the Haar Cascade face detection model
face_cascade = cv.CascadeClassifier(  # Create a CascadeClassifier object
    cv.samples.findFile(cv.data.haarcascades + 'haarcascade_frontalface_default.xml'))


# for output image
font = ImageFont.truetype("/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf", 36)

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


def pose_and_gesture(frame, frame_number):
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    sign_recognizer.recognize(mp_image)
    pose_results = pose.process(frame_rgb)
    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame

def facial_expression(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    try:
        result = DeepFace.analyze(frame, actions=['emotion'])
        dominant_emotion = result[0]['dominant_emotion']
    except ValueError:
        dominant_emotion = "No emotion detected"

    return dominant_emotion

class CurrentState():
    def __init__(self):
        self.facial_expression = None
        self.pose_and_gesture = None

current_state = CurrentState()

def update_webcam():
    return cam.read()

cam_read_success, last_frame_read = update_webcam()
last_facial_result = None
last_gesture_pose_result = None
frame_number = 0
def update_output_image():
    output_image = Image.new('RGB', (screen_width, screen_height), color = (153, 153, 255))
    out_img_draw = ImageDraw.Draw(output_image)

    if last_gesture_pose_result is not None:
        output_image.paste(Image.fromarray(last_gesture_pose_result), box=(0, 0))
    else:
        "No pose"

    if last_facial_result is not None:
        out_img_draw.text((10, 10), last_facial_result, font=font)

    return output_image # This will open an independent window



with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = {
            executor.submit(update_webcam) : "cam_image",
            executor.submit(facial_expression, last_frame_read) : "facial_expression",
            executor.submit(pose_and_gesture, frame=last_frame_read, frame_number=frame_number) : "gesture_and_pose"
        }

        while futures:
            completed_tasks, not_completed_tasks = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for task in completed_tasks:
                task_name = futures.pop(task)

                cam_read_success, last_frame_read = update_webcam()
                frame_number += 1
                print(f"Updating webcam has been successful - {cam_read_success}")


                if task_name == "facial_expression":
                    last_facial_result = task.result()
                    print(f"The last facial expression read was {last_facial_result}")


                elif task_name == "gesture_and_pose":
                    last_gesture_pose_result = task.result()
                    print(f"The last pose result was  {frame_number}")



                output_image = update_output_image()
                cv.imshow("Demo", np.asarray(output_image))

                if cv.waitKey(1) & 0xFF == ord('q'):  # quit when 'q' is pressed
                    cam.release()
                    break

            running_tasks = [futures[a] for a in not_completed_tasks]
            print(running_tasks)
            if "facial_expression" not in running_tasks:
                next_worker = executor.submit(facial_expression, last_frame_read)
                futures[next_worker] = "facial_expression"
            if "gesture_and_pose" not in running_tasks:
                next_worker = executor.submit(pose_and_gesture, frame=last_frame_read, frame_number=frame_number)
                futures[next_worker] = "gesture_and_pose"

