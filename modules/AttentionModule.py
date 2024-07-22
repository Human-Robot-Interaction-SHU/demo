import cv2 as cv
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from modules.Models import EyeTrackingForEveryone


class AttentionModule:
    """
    Takes an image
    """
    def __init__(self):
        model = EyeTrackingForEveryone()
        model.load_state_dict(torch.load("./weights/attention_weights", map_location=torch.device('cpu')))
        gaze_model = model.to('cpu')

        self.device = 'cpu'
        self.gaze_model = gaze_model
        self.base_options = python.BaseOptions("./weights/blaze_face_short_range.tflite")
        self.options = vision.FaceDetectorOptions(base_options=self.base_options)
        self.detector = vision.FaceDetector.create_from_options(self.options)

        self.BaseOptions = mp.tasks.BaseOptions

        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.lm_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path="./weights/face_landmarker.task"),
            running_mode=self.VisionRunningMode.IMAGE)

        self.fl_detector = self.FaceLandmarker.create_from_options(self.lm_options)

        self.bin_mask_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((25, 25)),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.eye_face_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((244, 244))
            #, IS THIS THE PROBLEM
            #v2.ToDtype(torch.float32)
        ])
        self.last_left_eye = None
        self.last_right_eye = None
        self.last_face = None
        self.last_binary_mask = None

    def get_eyes_and_face(self, cv_img):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_img)
        res = self.fl_detector.detect(img)
        try:
            right_eye_x2 = int(res.face_landmarks[0][33].x * cv_img.shape[1])
            right_eye_x1 = int(res.face_landmarks[0][133].x * cv_img.shape[1])
            right_eye_y1 = int(res.face_landmarks[0][33].y * cv_img.shape[0])
            right_eye_y2 = int(res.face_landmarks[0][133].y * cv_img.shape[0])
            right_eye_width = right_eye_x1 - right_eye_x2
            right_eye = cv_img[right_eye_y1 - right_eye_width // 2:right_eye_y1 + right_eye_width // 2,
                        right_eye_x2: right_eye_x1]
            self.last_right_eye = right_eye

            left_eye_x2 = int(res.face_landmarks[0][362].x * cv_img.shape[1])
            left_eye_x1 = int(res.face_landmarks[0][263].x * cv_img.shape[1])
            left_eye_y1 = int(res.face_landmarks[0][263].y * cv_img.shape[0])
            left_eye_y2 = int(res.face_landmarks[0][362].y * cv_img.shape[0])
            left_eye_width = left_eye_x1 - left_eye_x2

            left_eye = cv_img[left_eye_y1 - left_eye_width // 2:left_eye_y1 + left_eye_width // 2, left_eye_x2: left_eye_x1]
        except:
            # TODO this is added to allow training before fixing, needs removing later
            return False, False, False, False

        self.last_left_eye = left_eye
        detection_result = self.detector.detect(img)
        try:
            face_top = detection_result.detections[0].bounding_box.origin_y
            face_bottom = detection_result.detections[0].bounding_box.origin_y + detection_result.detections[
                0].bounding_box.height
            face_left = detection_result.detections[0].bounding_box.origin_x
            face_right = detection_result.detections[0].bounding_box.origin_x + detection_result.detections[
                0].bounding_box.width
        except IndexError:
            # TODO this is added to allow training before fixing, needs removing later

            return False, False, False, False

        face = cv_img[face_top:face_bottom, face_left:face_right]
        self.last_face = face
        binary_mask = np.zeros(cv_img.shape[:2])
        binary_mask[face_top:face_bottom, face_left:face_right] = 1

        self.last_binary_mask = binary_mask*255
        return left_eye, right_eye, face, binary_mask

    def get_x_y_from_image(self, img):
        bin_mask_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((25,25)),
                v2.ToDtype(torch.float32, scale=True)
            ])
        eye_face_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((244,244)),
            ])
        try:
            left_eye, right_eye, face, face_mask = self.get_eyes_and_face(img)
            left_eye = eye_face_transform(left_eye)
            right_eye = eye_face_transform(right_eye)
            face = eye_face_transform(face)

            face_mask = bin_mask_transform(Image.fromarray(face_mask * 255))
        except AttributeError:
            return False, False, False, False
        except RuntimeError:
            return False, False, False, False
        if left_eye is not False:
            return self.gaze_model(left_eye.float().to(self.device).unsqueeze(0), right_eye.float().to(self.device).unsqueeze(0), face.float().to(self.device).unsqueeze(0), face_mask.float().to(self.device).unsqueeze(0)).cpu().detach().numpy()
        else:
            return False, False, False, False
