from application.body_pose.PoseDetectionModule import PoseDetectionModule
import cv2 as cv
from PIL import Image, ImageDraw
import numpy as np
import mediapipe as mp


class DisplayModule:

    def __init__(self, cam_display_mod):
        self.cam_display = cam_display_mod
        self.out_img = Image.new('RGB', (self.cam_display.screen_width, self.cam_display.screen_height), color=(153, 153, 255))
        self.out_img_draw = ImageDraw.Draw(self.out_img)

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
            # out_img_draw.arc([(attention_out[0][0] - 25, attention_out[0][1] - 25),
            #                 (attention_out[0][0] + 25, attention_out[0][1] + 25)], start=0, end=360,
            #                 fill=(0, 0, 255))
            out_img_draw.ellipse([(attention_out[0][0] - 25, attention_out[0][1] - 25),
                                  (attention_out[0][0] + 25, attention_out[0][1] + 25)],
                                 fill=(0, 255, 0), outline=None)

            out_img_draw.text((1000, 10), f"Gaze location : {int(attention_out[0][0])}, {int(attention_out[0][1])}",
                              font=self.cam_display.font)
            #print(attention_out[0][0])

    def draw_face_result(self, dominant_emotion, out_img_draw):
        out_img_draw.text((10, 10), f"Face: {dominant_emotion}", font=self.cam_display.font)

    def draw_tone_result(self, tone=None):
        self.out_img_draw.text((self.cam_display.screen_width - 300, self.cam_display.screen_height - 200),
                               "Tone: " + tone if tone else "Tone: -- ", font=self.cam_display.font)


    def draw_background_boxes_for_texts(self):
        screen_width = self.cam_display.screen_width
        screen_height = self.cam_display.screen_height
        box_height = 150

        # Create a semi-transparent black box
        box_img = Image.new('RGBA', (screen_width, box_height), (0, 0, 0, 128))  # 128 for 50% opacity
        box_img_bottom = Image.new('RGBA', (screen_width, 200), (0, 0, 0, 128))  # 128 for 50% opacity
        top_box_position = (0, 0)
        bottom_box_position = (0, screen_height - box_height - 80)

        # Paste the boxes onto the output image
        self.out_img.paste(box_img, top_box_position, box_img)
        self.out_img.paste(box_img_bottom, bottom_box_position, box_img_bottom)

    def display_results(self, img, video_results, audio_result):
        self.draw_pose_result_on_image(img, video_results["pose"].pose_landmarks, self.out_img)

        self.draw_background_boxes_for_texts()

        self.draw_tone_result()

        self.out_img_draw.text((10, self.cam_display.screen_height - 200), "Content of speech", font=self.cam_display.font)

        if audio_result:
            audio_text = audio_result[1] if audio_result[1] else " - - "
            formatted_audio_result = f"{audio_result[2]}: {audio_text}"

            self.out_img_draw.text(
                (10, self.cam_display.screen_height - 150),
                formatted_audio_result,
                font=self.cam_display.font
            )

        self.draw_attention_result(video_results["attention"], self.out_img_draw)
        self.draw_face_result(video_results["face"], self.out_img_draw)

        cv.imshow("Demo", np.asarray(self.out_img))  # This will open an independent window
