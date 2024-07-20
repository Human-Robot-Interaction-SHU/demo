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
frame_number = 0

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

while True:
    result, image = cam.read()

    # tracking frame number for pose detection
    frame_number += 1

    # output image
    out_img = Image.new('RGB', (screen_width, screen_height), color = (153, 153, 255))
    out_img_draw = ImageDraw.Draw(out_img)
    # image successfully read from webcam
    if result:

        # attention
        attention_out = attn.get_x_y_from_image(image)
        # attention can fail if no face detected
        if attention_out[0] is not False:

            # Draw circle where gaze detected
            out_img_draw.arc([(attention_out[0][0]-25, attention_out[0][1]-25),(attention_out[0][0]+25, attention_out[0][1]+25)], start=0, end=360, fill=(255, 255, 255))
            out_img_draw.text((1000, 10), f"Gaze location : {int(attention_out[0][0])}, {int(attention_out[0][1])}", font=font)
            print(attention_out[0][0])

        # body pose
        frame_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        sign_recognizer.recognize_async(mp_image, frame_number)
        pose_results = pose.process(frame_rgb)
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # facial expression
        # Convert the image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray)

        # Check if any faces are detected
        if len(faces) > 0:
            try:
                # Analyze the image for emotions
                analysis_result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis_result[0]['dominant_emotion']
                print(f"Dominant Emotion: {dominant_emotion}")
            except ValueError as ve:
                print(f"ValueError: {ve}")
                dominant_emotion = "No emotion detected"
            except Exception as e:
                print(f"An error occurred: {e}")
                dominant_emotion = "No emotion detected"
        else:
            print("No faces detected in the image.")
            dominant_emotion = "No emotion detected"

        # Display the dominant emotion on the image
        if dominant_emotion != "No emotion detected":
            out_img_draw.text((10, 10), dominant_emotion, font=font)
    else:
        print("Failed to get image from webcam")

    out_img.paste(Image.fromarray(image), box=(0,0))

    cv.imshow("Demo", np.asarray(out_img)) # This will open an independent window

    if cv.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
        cam.release()
        break


cam.release()
cv.destroyAllWindows()
