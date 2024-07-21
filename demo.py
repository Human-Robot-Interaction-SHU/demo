import asyncio

from modules.PoseDetection import PoseDetectionModule
from modules.content_of_speech_emotion_recognizer import run_audio_detection
from modules.models import EyeTrackingForEveryone
from modules.AttentionModule import AttentionModule
import torch
import cv2 as cv
from screeninfo import get_monitors
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import mediapipe as mp
import os
from concurrent.futures import ThreadPoolExecutor

# Facial recognition
from deepface import DeepFace

class FaceDetectionModule:

    face_cascade = cv.CascadeClassifier(  # Create a CascadeClassifier object
            cv.samples.findFile(cv.data.haarcascades + 'haarcascade_frontalface_default.xml'))

    def detect_emotion(self, img):
        return DeepFace.analyze(img, actions=['emotion'])  # , enforce_detection=False)


class CameraAndMonitorModule:
    def __init__(self):
        try:
            monitor = get_monitors()
            self.screen_width = monitor[0].width
            self.screen_height = monitor[0].height
        except:
            print("No monitors found")
            self.screen_width = 1920
            self.screen_height = 1080

        # Initialize the camera
        self.cam = cv.VideoCapture(0)
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        # Retrieve and print the FPS value
        self.fps = self.cam.get(cv.CAP_PROP_FPS)
        print(f"Camera FPS: {self.fps}")

        self.font = ImageFont.load_default(36)

    def __del__(self):
        # Release the camera when the object is deleted
        if self.cam.isOpened():
            self.cam.release()


class VideoEmotionsRecognizer:
    def __init__(self, attn_mod, pose_mod, face_mod):
        self.attn = attn_mod
        self.pose_module = pose_mod
        self.face_module = face_mod


    def get_emotion_results_from_models(self, image, frame_number):
        emotions = {
            "pose": None,
            "attention": None,
            "face": None
        }

        def process_pose(img, frm_num):
            body_pose_image = img.copy()
            frame_rgb = cv.cvtColor(body_pose_image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self.pose_module.sign_recognizer.recognize_async(mp_image, frm_num)
            pose_results = self.pose_module.pose.process(frame_rgb)
            return pose_results

        def process_attention(img):
            attention_out = self.attn.get_x_y_from_image(img)
            return attention_out

        def process_face(img):
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            faces = self.face_module.face_cascade.detectMultiScale(gray)
            if len(faces) > 0:
                try:
                    analysis_result = self.face_module.analyze(img)
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

        with ThreadPoolExecutor() as executor:
            pose_future = executor.submit(process_pose, image, frame_number)
            attention_future = executor.submit(process_attention, image)
            face_future = executor.submit(process_face, image)

            emotions["pose"] = pose_future.result()
            emotions["attention"] = attention_future.result()
            emotions["face"] = face_future.result()

        return emotions


class DisplayModule:

    def __init__(self, cam_display_mod):
        self.cam_display = cam_display_mod

    def draw_pose_result_on_image(self, img, pose_landmarks, out_img):
        img = img.copy()
        mp.solutions.drawing_utils.draw_landmarks(img, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        image_height, image_width, _ = img.shape
        cx, cy = PoseDetectionModule.handPosition.x * image_width, PoseDetectionModule.handPosition.y * image_height

        frame = cv.putText(img=img, text=PoseDetectionModule.handSign, org=(int(cx), int(cy)), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255, 0, 0), thickness=3)

        resized_image = cv.resize(img, (self.cam_display.screen_width, self.cam_display.screen_height), interpolation=cv.INTER_AREA)
        resized_image_pil = Image.fromarray(resized_image)
        out_img.paste(resized_image_pil, (0, 0))

    def draw_attention_result(self, attention_out, out_img_draw):
        if attention_out[0] is not False:
            out_img_draw.arc([(attention_out[0][0] - 25, attention_out[0][1] - 25),
                            (attention_out[0][0] + 25, attention_out[0][1] + 25)], start=0, end=360,
                            fill=(255, 255, 255))
            out_img_draw.text((1000, 10), f"Gaze location : {int(attention_out[0][0])}, {int(attention_out[0][1])}",
                            font=self.cam_display.font)
            print(attention_out[0][0])

    def draw_face_result(self, dominant_emotion, out_img_draw):
        out_img_draw.text((10, 10), dominant_emotion, font=self.cam_display.font)

    def display_results(self, img, results):
        out_img = Image.new('RGB', (self.cam_display.screen_width, self.cam_display.screen_height), color=(153, 153, 255))
        out_img_draw = ImageDraw.Draw(out_img)

        self.draw_pose_result_on_image(img, results["pose"].pose_landmarks, out_img)
        self.draw_attention_result(results["attention"], out_img_draw)
        self.draw_face_result(results["face"], out_img_draw)

        cv.imshow("Demo", np.asarray(out_img))  # This will open an independent window


async def run_audio():
    await run_audio_detection()


# Function to run an async task in a thread
def run_async_in_thread(async_func):
    return asyncio.run(async_func)


async def run_both():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        # Schedule async tasks to run in separate threads
        task1_future = loop.run_in_executor(executor, run_async_in_thread, run_audio())
        task2_future = loop.run_in_executor(executor, run_async_in_thread, run_video_detection())

        # Wait for both tasks to complete (they won't in this case, as they run indefinitely)
        await asyncio.gather(task1_future, task2_future)


async def run_video_detection():
    attn = AttentionModule()
    pose_module = PoseDetectionModule('./weights/gesture_recognizer.task')
    face_module = FaceDetectionModule()
    cam_monitor = CameraAndMonitorModule()
    video_emotions_recognizer = VideoEmotionsRecognizer(attn, pose_module, face_module)
    display = DisplayModule(cam_monitor)

    frame_number = 0

    while cam_monitor.cam.isOpened():
        result, image = cam_monitor.cam.read()

        # tracking frame number for pose detection
        frame_number += 1

        # image successfully read from webcam
        if result:
            res = video_emotions_recognizer.get_emotion_results_from_models(image, frame_number)
            display.display_results(image, res)
        else:
            print("Failed to get image from webcam")

        if cv.waitKey(1) & 0xFF == ord('q'):  # quit when 'q' is pressed
            del cam_monitor
            break

    del cam_monitor
    cv.destroyAllWindows()

if __name__ == "__main__":

    asyncio.run(run_both())


