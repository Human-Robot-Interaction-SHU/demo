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
import os

import concurrent.futures

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
model.load_state_dict(torch.load("./weights/attention_weights", map_location=torch.device('cpu')))
attn = AttentionModule(model.to('cpu'))

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

model_path = './weights/gesture_recognizer.task'
with open(model_path, 'rb') as file:
    model_data = file.read()

base_options = BaseOptions(model_asset_buffer=model_data)
VisionRunningMode = mp.tasks.vision.RunningMode
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                          result_callback=setResults)
sign_recognizer = vision.GestureRecognizer.create_from_options(options)


# end pose detection


# facial expression

# initialize the Haar Cascade face detection model
face_cascade = cv.CascadeClassifier(  # Create a CascadeClassifier object
    cv.samples.findFile(cv.data.haarcascades + 'haarcascade_frontalface_default.xml'))


# for output image
font = ImageFont.load_default(36)

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# Retrieve and print the FPS value
fps = cam.get(cv.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")

def get_emotion_results_from_models(image, frame_number):
    emotions = {
        "pose": None,
        "attention": None,
        "face": None
    }

    def process_pose(img, frm_num):
        body_pose_image = img.copy()
        frame_rgb = cv.cvtColor(body_pose_image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        sign_recognizer.recognize_async(mp_image, frm_num)
        pose_results = pose.process(frame_rgb)
        return pose_results

    def process_attention(img):
        attention_out = attn.get_x_y_from_image(img)
        return attention_out

    def process_face(img):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        if len(faces) > 0:
            try:
                analysis_result = DeepFace.analyze(img, actions=['emotion'])#, enforce_detection=False)
                dominant_emotion = analysis_result[0]['dominant_emotion']
            except ValueError as ve:
                print(f"ValueError: {ve}")
                dominant_emotion = "No emotion detected"
            except Exception as e:
                print(f"An error occurred: {e}")
                dominant_emotion = "No emotion detected"
        else:
            print("No faces detected in the image.")
            dominant_emotion = "No emotion detected"
        return dominant_emotion

    with concurrent.futures.ThreadPoolExecutor() as executor:
        pose_future = executor.submit(process_pose, image, frame_number)
        attention_future = executor.submit(process_attention, image)
        face_future = executor.submit(process_face, image)

        emotions["pose"] = pose_future.result()
        emotions["attention"] = attention_future.result()
        emotions["face"] = face_future.result()

    return emotions


def draw_pose_result_on_image(img, pose_landmarks, out_img):
    img = img.copy()
    mp_drawing.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Get the image as a PIL image and resize it to fill the screen
    # Resize the image using OpenCV
    resized_image = cv.resize(img, (screen_width, screen_height), interpolation=cv.INTER_AREA)

    # Convert the resized image to a PIL image
    resized_image_pil = Image.fromarray(resized_image)

    # Paste the resized image onto the output image
    out_img.paste(resized_image_pil, (0, 0))


def draw_attention_result(attention_out, out_img_draw):
    #attention can fail if no face detected
    if attention_out[0] is not False:
        # Draw circle where gaze detected
        out_img_draw.arc([(attention_out[0][0] - 25, attention_out[0][1] - 25),
                          (attention_out[0][0] + 25, attention_out[0][1] + 25)], start=0, end=360,
                         fill=(255, 255, 255))
        out_img_draw.text((1000, 10), f"Gaze location : {int(attention_out[0][0])}, {int(attention_out[0][1])}",
                          font=font)
        print(attention_out[0][0])


def draw_face_result(dominant_emotion, out_img_draw):
    out_img_draw.text((10, 10), dominant_emotion, font=font)


def detect_image_emotion(img, frame_num):

    out_img = Image.new('RGB', (screen_width, screen_height), color = (153, 153, 255))
    out_img_draw = ImageDraw.Draw(out_img)

    results = get_emotion_results_from_models(img, frame_num)

    draw_pose_result_on_image(img, results["pose"].pose_landmarks, out_img)
    draw_attention_result(results["attention"], out_img_draw)
    draw_face_result(results["face"], out_img_draw)

    cv.imshow("Demo", np.asarray(out_img))  # This will open an independent window



frame_number = 0

while True:
    result, image = cam.read()

    # tracking frame number for pose detection
    frame_number += 1

    # image successfully read from webcam
    if result:
        detect_image_emotion(image, frame_number)
    else:
        print("Failed to get image from webcam")

    if cv.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
        cam.release()
        break


cam.release()
cv.destroyAllWindows()
