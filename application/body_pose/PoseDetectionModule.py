import os.path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
# Get the directory of the current script
script_dir = os.path.dirname(script_path)


class PoseDetectionModule:
    handSign = ""
    handPosition = NormalizedLandmark(x=0.5, y=0.5)

    def __init__(self):
        # Initialize the pose detector
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        model_path = os.path.join(script_dir, 'weights/gesture_recognizer.task')

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
