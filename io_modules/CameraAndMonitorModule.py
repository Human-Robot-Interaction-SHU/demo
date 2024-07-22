from screeninfo import get_monitors
import cv2 as cv
from PIL import ImageFont


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
