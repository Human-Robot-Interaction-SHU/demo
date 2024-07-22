import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult


class PoseDetectionModule:
    handSign = ""
    handPosition = NormalizedLandmark(x=0.5, y=0.5)

    def __init__(self, model_path):
        # Initialize the pose detector
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Load gesture recognizer model

        with open(model_path, 'rb') as file:
            model_data = file.read()

        # Configure gesture recognizer options
        base_options = BaseOptions(model_asset_buffer=model_data)
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.setResults
        )

        # Create gesture recognizer
        self.sign_recognizer = vision.GestureRecognizer.create_from_options(options)

    @staticmethod
    def setResults(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        if result.gestures and result.gestures[0]:
            PoseDetectionModule.handSign = result.gestures[0][0].category_name
            PoseDetectionModule.handPosition = result.hand_landmarks[0][0]
