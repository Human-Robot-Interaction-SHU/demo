from modules.models import EyeTrackingForEveryone
from modules.AttentionModule import AttentionModule
import torch
import cv2 as cv
from screeninfo import get_monitors
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Pose detection
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult
from mediapipe.tasks.python import BaseOptions, vision
import mediapipe as mp
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
#model.load_state_dict(torch.load("./horizontal_flipped_epochs_19"))
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
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                          result_callback=setResults)
sign_recognizer = vision.GestureRecognizer.create_from_options(options)
frame_number = 0

# end pose detection


cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)



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

        # body pose
        frame_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        sign_recognizer.recognize_async(mp_image, frame_number)
        pose_results = pose.process(frame_rgb)
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        print("Failed to get image from webcam")

    out_img.paste(Image.fromarray(image), box=(0,0))

    cv.imshow("Demo", np.asarray(out_img)) # This will open an independent window

    if cv.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
        cam.release()
        break


cam.release()
cv.destroyAllWindows()
