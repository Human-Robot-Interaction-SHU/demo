import asyncio
from application.VideoEmotionsRecognizer import VideoEmotionsRecognizer
from application.body_pose.PoseDetectionModule import PoseDetectionModule
from application.content_of_speech.ContentOfSpeechModule import ContentOfSpeechEmotionRecognizer
from application.attention.AttentionModule import AttentionModule
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from application.face_analysis.FaceDetectionModule import FaceDetectionModule
from io_modules.DisplayModule import DisplayModule
from io_modules.camera_and_monitor_module import CameraAndMonitorModule

content_of_speech = ContentOfSpeechEmotionRecognizer()


# Function to run an async task in a thread
def run_async_in_thread(async_func):
    try:
        return asyncio.run(async_func)
    except Exception as e:
        print(f"Exception caught in thread: {e}")


async def execute_tasks_in_thread_async():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        # Schedule async tasks to run in separate threads
        speech_task = loop.run_in_executor(executor, run_async_in_thread, content_of_speech.run())
        video_task = loop.run_in_executor(executor, run_async_in_thread, run_video_detection())

        # Wait for both tasks to complete (they won't in this case, as they run indefinitely)
        await asyncio.gather(video_task, speech_task)


async def run_video_detection():
    cam_monitor = CameraAndMonitorModule()
    display = DisplayModule(cam_monitor)

    attn = AttentionModule()
    pose_module = PoseDetectionModule()
    face_module = FaceDetectionModule()

    video_emotions_recognizer = VideoEmotionsRecognizer(attn, pose_module, face_module)

    frame_number = 0

    while cam_monitor.cam.isOpened():
        result, image = cam_monitor.cam.read()

        # tracking frame number for pose detection
        frame_number += 1

        # image successfully read from webcam
        if result:
            res = video_emotions_recognizer.get_emotion_results_from_models(image, frame_number)
            audio_emo = content_of_speech.emotion_results[len(content_of_speech.emotion_results)-1] \
                if len(content_of_speech.emotion_results) > 0 else None
            display.display_results(image, res, audio_emo)

        else:
            print("Failed to get image from webcam")

        if cv.waitKey(1) & 0xFF == ord('q'):  # quit when 'q' is pressed
            del cam_monitor
            break

    del cam_monitor
    cv.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(execute_tasks_in_thread_async())
