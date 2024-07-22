from deepface import DeepFace
import cv2 as cv

class FaceDetectionModule:

    face_cascade = cv.CascadeClassifier(  # Create a CascadeClassifier object
            cv.samples.findFile(cv.data.haarcascades + 'haarcascade_frontalface_default.xml'))

    def detect_emotion(self, img):
        return DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
