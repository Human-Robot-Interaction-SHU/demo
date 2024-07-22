import cv2 as cv
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor


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
                    analysis_result = self.face_module.detect_emotion(img)
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

