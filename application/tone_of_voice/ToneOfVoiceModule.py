from transformers import pipeline
import torch
import librosa
import io
import pyaudio
from pydub import AudioSegment

from application.SharedConstants import AUDIO_SAMPLE_RATE


class ToneOfVoiceModule:
    def __init__(self):
        # Set up the device and model pipeline
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "mps") if torch.backends.mps.is_available() else torch.device("cpu")

        self.classifier = pipeline("audio-classification",
                                   model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                                   device=self.device)

        self.emotion_results = []

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None

    def create_wav(self, audio_bytes):
        audio = AudioSegment(
            data=audio_bytes,
            sample_width=2,  # 2 bytes for 16-bit audio
            frame_rate=AUDIO_SAMPLE_RATE,
            channels=1  # Mono audio
        )
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io

    def process_audio(self, audio_bytes):
        #print("Processing audio for emotion prediction...")

        # Convert audio bytes to WAV file-like object
        audio_file = self.create_wav(audio_bytes)
        audio_file.seek(0)

        # Load the audio using librosa
        audio, sr = librosa.load(audio_file, sr=AUDIO_SAMPLE_RATE)

        # Make the prediction
        result = self.classifier(audio)

        main_emotion = sorted(result, key=lambda x: x['score'], reverse=True)[0]

        #print(f"Top Emotion: {main_emotion['label']}, Score: {main_emotion['score']:.4f}")

        self.emotion_results.append(main_emotion['label'])