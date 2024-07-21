import sys
import re
import pyaudio
from transformers import pipeline
import torch
import librosa
import queue
import io
from pydub import AudioSegment


def create_wav(audio_bytes):
    audio =  AudioSegment(
        data=audio_bytes,
        sample_width=2,  # 2 bytes for 16-bit audio
        frame_rate=16000,
        channels=1  # Mono audio
    )
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io


# def create_wav(frames):
#     output = io.BytesIO()
#     with wave.open(output, "wb") as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(44100)
#         wf.writeframes(b''.join(frames))
#     output.seek(0)  # Rewind the BytesIO object to the beginning
#     return output


def predict_emotion(audio_stream):
    # Check if MPS is available
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


    # Load the pretrained model
    classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                      device=device)

    # Load your audio file
    audio_file = create_wav(audio_stream)
    # "/Users/valentineo/PycharmProjects/tone-emo/src/data/question.wav"

    # Ensure the audio_file is rewound to the beginning
    audio_file.seek(0)

    # Load audio using librosa
    audio, sr = librosa.load(audio_file, sr=16000)

    # Perform emotion classification
    result = classifier(audio)

    # Print the results
    for emotion in result:
        print(f"Emotion: {emotion['label']}, Score: {emotion['score']:.4f}")

    # Sort the results by score in descending order and get the top emotion
    main_emotion = sorted(result, key=lambda x: x['score'], reverse=True)[0]

    # Print only the top emotion
    print(f"Top Emotion: {main_emotion['label']}, Score: {main_emotion['score']:.4f}")

    return main_emotion['label']


RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate=16000, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        self.audio_data = b''

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
        self.audio_data += in_data
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

