from threading import Thread
import asyncio
import queue
import sys
import re
from google.cloud import speech
from google.oauth2 import service_account
import pyaudio
import threading

from application.SharedConstants import AUDIO_SAMPLE_RATE
from application.content_of_speech.ContentOfSpeechModule import ContentOfSpeechEmotionRecognizer
from application.tone_of_voice.ToneOfVoiceModule import ToneOfVoiceModule


class AudioEmotionRecognizer:

    def __init__(self):
        self.content_of_speech = ContentOfSpeechEmotionRecognizer()
        self.tone_of_voice_module = ToneOfVoiceModule()  # Instantiate ToneOfVoiceModule

    async def run(self):

        def run_speech():
            # Initialize MicrophoneStream
            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
                self.content_of_speech.run(audio_generator)
        def run_tone():

            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
                accumulated_chunks = []
                for chunk in audio_generator:
                    accumulated_chunks.append(chunk)
                    if len(accumulated_chunks) >= 79:
                        # Combine accumulated chunks into one byte array
                        audio_data = b"".join(accumulated_chunks)
                        res = self.tone_of_voice_module.process_audio(audio_data)  # Process the accumulated audio
                        print("Tone: ", res)
                        accumulated_chunks = []  # Reset the accumulated chunks

        thread1 = threading.Thread(target=run_speech)
        thread2 = threading.Thread(target=run_tone)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()



# Audio recording parameters

CHUNK = int(AUDIO_SAMPLE_RATE / 10)  # 100ms

# Global list to store emotion data

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate=AUDIO_SAMPLE_RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

