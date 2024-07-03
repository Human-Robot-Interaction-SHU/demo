import os
import asyncio
import wave
import shutil
from datetime import datetime
import json

from transformers import (
    AutoTokenizer
)
from .speech_content_model import BertForMultiLabelClassification, MultiLabelPipeline

from .google_speech_to_text import StreamingTranscriber

tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
pipeline = MultiLabelPipeline(model=model, tokenizer=tokenizer, threshold=0.3)

service_account_info = {
  "type": "service_account",
  "project_id": "human-robot-interaction-427202",
  "private_key_id": "5a8fa3a319d052106f64b5a857148b54784b9a18",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDGLRnNSECMfO3E\nuLaJswQYwK/v5N9rQL39TA/SDel/47cBcAb9n2CEmRdH8RILj33wpJ/ccwUrA5df\nop/HoPzFynPPXcQtNC3HEi3wrOGxPlPI75pL2aUE34jb0LngtqfGUstfD+trphQt\nWlFvczFcHXwgEv1N64qkXYFNid8isAp7IldZdtjRwKFSf4aNCvRGiUrARMWhRAMc\nYhL1VypxKBGHM3Qhsov7RU+wt8GGTWLYbeup64pdLT//U0B6vaTn72bTRq+7c0bE\nrKhzFEeepCEwzzJ25mIIBVvPserBqxpikWuwvZOxzuykpBVifuWgRcikcG0g0K4x\nFGb1icFxAgMBAAECggEAXgnHvRgkfSXJA/jssYHPl1lUAz91XyUNIpWFylTMsOGj\nFR0OTCplN/aXTA2SVQcFqXvM2eSAls0w9vIp5KY5XDf55XQmo5anhFfVkefPfvZG\n9snvyz9fZWUXQcuVcJLsIRlnpNfejCn2WCEMFJkyWnYpUOUB6wgytVUjhuI+Dmxp\n2tsoIMh6F3fadWoIQy/4AoqZ6hq+Yi5HMHpcUFC/1Qf+erYcZ2hJbRsFM+PoGGQ4\neHli1gu9XiUlAViglyWLh5o88G6LQrApQBEmrBW585u6zxwo6NNQibHeVGuvpzlA\ngnY54Pm2s7pJXrwyrhz9tv7BmfElfbAnh1i594NIewKBgQDp0vKonqE2tg3a76W6\n1ZEbwe9E0uEMzHqHRJyOnuuDxaoByZQlD0rzrM3L/fhZFnITWRwtOys2CoC0uthx\nds4aHkLAsZcuXEmlgeI696JU4OmWHuSeIphLF5h0iBjmnEs0P7IYHWTMzEG+Q38x\nJoPRpJNZzfRn6+ggvZOvT08yRwKBgQDY+Kcvbp1Jbz2FLt/j2dFMyQ2coqiyf1N/\nNBeMtDCNj21a0GPrN/xiSiQ0eqzWPniKhtk8As0xRLAZVXcVkI9u433YmZqhu8r1\nmVqrV9KGT9GhW/RRTiRAhtTUwWjEtBNVApGEAq2wB2fCE4EaGfHNV1fT4gPr5RlO\n8CTmdDjShwKBgFG8MKDq2pXia9N1ZCx8TT4zu60GPi8YJ1izjjp4qQEmDniTe1q9\nDslBRasiOzcBFp1Wz/ersD4yy6zhh5maGw+cNl9fdOZ60i+tyGQufitHd7/HSslQ\ndIYDWIKbtICgb9Vy0pGFbN/+IpkcxRBsUzXsXqnMybuuBjWzrzVf9uIvAoGAXd4N\nbl7bm0aOBg2GfSvh+edNhUN12mttcy3VNmFKVCQF+nEHmV7KSLesvCuKlNHIEp5O\nY0EPBs6hpQQtld3Jv/6Zlli15ly5bNGgwVooUUU8+yMuKvK0iloKv9TA/8CsUG3h\nCIykGfDKOdN4WhN5Yg30iE1Sxv6BmX4ZaL5FSwcCgYAQTPKGqqEfsW2oDjcZfcGY\nentgPBuHmyA1u621zBUpHC5GH9HOxnbFFAolED6DaEqaZpjYQ9OrPtsiFklxEBGY\nUM610x45TB1wHvetaRhYauyXXtaZC+UofOuSk+mfSoxlvwCsn5lrFLfY7dXzA3Va\nNA9sLichxePueEovOO1ZDA==\n-----END PRIVATE KEY-----\n",
  "client_email": "speech-human-robot-interation@human-robot-interaction-427202.iam.gserviceaccount.com",
  "client_id": "108185768289001893274",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/speech-human-robot-interation%40human-robot-interaction-427202.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}



class ContentOfSpeechEmotionRecognizer:
    def __init__(self):
        self.audio_buffer = b''
        self.buffer_size_limit = 1024 * 1024 * 0.25 # 1 MB limit for buffer
        self.buffer_timeout = 10  # 20 seconds timeout for sending buffer
        self.session_id = None
        self.session_folder = None
        self.buffer_count = 0
        self.send_task = None
        self.session_audio_filename = None
        self.session_transcription_filename = None
        self.audio_chunks = []

        # Initialize tokenizer, model, and pipeline
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
        self.model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
        self.pipeline = MultiLabelPipeline(model=self.model, tokenizer=self.tokenizer, threshold=0.3)
        self.streaming_transcriber = StreamingTranscriber(service_account_info=service_account_info)
        self.initialize_session()

    def initialize_session(self):
        # Generate a unique session ID (you can use a timestamp or any unique identifier)
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.session_folder = os.path.join("session_data", self.session_id)
        os.makedirs(self.session_folder, exist_ok=True)
        self.session_audio_filename = os.path.join(self.session_folder, "session_audio.wav")
        self.session_transcription_filename = os.path.join(self.session_folder, "session_transcription.txt")

        # Create or open session audio file in write binary mode if it doesn't exist
        if not os.path.exists(self.session_audio_filename):
            with wave.open(self.session_audio_filename, 'wb') as session_audio_wf:
                session_audio_wf.setnchannels(1)
                session_audio_wf.setsampwidth(2)  # 2 bytes (16 bits)
                session_audio_wf.setframerate(16000)  # 48 kHz

    async def receive_audio_data(self, bytes_data):

        self.audio_buffer += bytes_data
        self.audio_chunks.append(bytes_data)
        print(len(self.audio_buffer))
        if len(self.audio_buffer) >= self.buffer_size_limit:
            return await self.process_buffer()
        # # else:
        # #     if not self.send_task:
        # #         self.send_task = asyncio.create_task(self.send_buffer_after_timeout())
        # return None

    async def send_buffer_after_timeout(self):
        try:
            await asyncio.sleep(self.buffer_timeout)
            return await self.process_buffer()
        except asyncio.CancelledError as e:
            print(f"send_buffer_after_timeout task was cancelled: {e}")
            return None


    async def process_buffer(self, write_to_storage=True):
        if not self.audio_buffer:
            return None

        audio_content = self.audio_buffer
        self.audio_buffer = b''
        if self.send_task:
            self.send_task.cancel()
            self.send_task = None

        if write_to_storage:
            buffer_folder = os.path.join(self.session_folder, f"buffer_{self.buffer_count}")
            os.makedirs(buffer_folder, exist_ok=True)

            audio_filename = os.path.join(buffer_folder, "audio.wav")
            temp_filename = "temp_audio.wav"

            # Create or open a temporary file to append new data
            with wave.open(temp_filename, 'wb') as temp_wf:
                temp_wf.setnchannels(1)
                temp_wf.setsampwidth(2)  # 2 bytes (16 bits)
                temp_wf.setframerate(48000)  # 48 kHz

                # Write existing audio file content to temporary file
                if os.path.exists(audio_filename):
                    with wave.open(audio_filename, 'rb') as original_wf:
                        temp_wf.writeframes(original_wf.readframes(original_wf.getnframes()))

                # Append new audio data
                temp_wf.writeframes(audio_content)

            # Replace the original file with the temporary file
            shutil.move(temp_filename, audio_filename)

        audio_generator = (chunk for chunk in self.audio_chunks)
        text = " ".join([t async for t in self.streaming_transcriber.transcribe_streaming(audio_generator)])

        if write_to_storage:
            transcription_filename = os.path.join(buffer_folder, "text.txt")
            with open(transcription_filename, 'w') as text_file:
                text_file.write(text)

        result = self.pipeline(text)
        for item in result:
            item['scores'] = [float(score) for score in item['scores']]
        json_response = {
            "transcription": text,
            "emotion_detection": result
        }
        json_result = json.dumps(json_response)

        self.buffer_count += 1
        self.audio_chunks = []

        if write_to_storage:
            # Append to session audio file in append binary mode
            with open(self.session_audio_filename, 'ab') as session_audio_file:
                session_audio_file.write(audio_content)

            # Append to session transcription file
            with open(self.session_transcription_filename, 'a') as session_transcription_file:
                session_transcription_file.write(text + "\n")

        return json_result
