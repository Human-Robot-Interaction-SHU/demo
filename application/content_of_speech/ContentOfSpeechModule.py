from threading import Thread
import asyncio
import queue
import sys
import re
from google.cloud import speech
from google.oauth2 import service_account
import pyaudio

from transformers import (
    Pipeline,
    PreTrainedTokenizer,
    ModelCard,
    PreTrainedModel,
    TFPreTrainedModel,
    BertPreTrainedModel,
    BertModel
)

from transformers.pipelines import ArgumentHandler
from typing import Union, Optional, List, Dict, Any
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline

class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MultiLabelPipeline(Pipeline):
    def __init__(
            self,
            model: Union[PreTrainedModel, TFPreTrainedModel],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            task: str = "",
            args_parser: ArgumentHandler = None,
            device: int = -1,
            binary_output: bool = False,
            threshold: float = 0.3
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task
        )
        self.threshold = threshold

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        return self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

    def _forward(self, model_inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.model(**model_inputs)

    def postprocess(self, model_outputs: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        if isinstance(model_outputs, tuple):
            model_outputs = model_outputs[0]

        logits = model_outputs.detach().numpy()
        scores = 1 / (1 + np.exp(-logits))  # Apply sigmoid

        results = []
        for item in scores:
            labels = []
            score_values = []
            for idx, s in enumerate(item):
                if s > self.threshold:
                    labels.append(self.model.config.id2label[idx])
                    score_values.append(s)
            results.append({"labels": labels, "scores": score_values})
        return results


class ContentOfSpeechEmotionRecognizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
        self.model = BertForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
        self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
        self.speech_to_text = GoogleSpeechToText()
        self.emotion_results = []

    def get_emotion(self, text):
        result = self.pipeline(text)
        emotions = [label['label'] for label in result[0] if label['score'] > 0.3]
        return text, emotions

    async def run(self):

        def emotion_handler(time_seconds, transcript):
            text, emotions = self.get_emotion(transcript)
            emotion_str = ", ".join(emotions)
            self.emotion_results.append((time_seconds, text, emotion_str))  # Store in global list

        # Initialize MicrophoneStream
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )
            responses = self.speech_to_text.client.streaming_recognize(self.speech_to_text.streaming_config, requests)
            stream.listen_print_loop(responses, emotion_handler)





google_config = {
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



# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# Global list to store emotion data

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate=16000, chunk=CHUNK):
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

    def listen_print_loop(self, responses, result_handler):
        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue
            result = response.results[0]
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            overwrite_chars = " " * (num_chars_printed - len(transcript))
            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + "\r")
                sys.stdout.flush()
                num_chars_printed = len(transcript)
            else:
                time_seconds = result.result_end_time.seconds + result.result_end_time.microseconds / 1e6

                # Call the provided result_handler function
                result_handler(time_seconds, transcript)

                if re.search(r"\b(exit|quit)\b", transcript, re.I):
                    print("Exiting..")
                    break
                num_chars_printed = 0

        return transcript


class GoogleSpeechToText:
    def __init__(self):
        self.language_code = "en-US"
        self.cred = service_account.Credentials.from_service_account_info(google_config)
        self.client = speech.SpeechClient(credentials=self.cred)
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=self.language_code,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config, interim_results=True
        )


