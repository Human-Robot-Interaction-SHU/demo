import sys
import re
from google.cloud import speech
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
from application.content_of_speech.GoogleSpeechText import GoogleSpeechToText


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

    def run(self, audio_generator):

        def emotion_handler(time_seconds, transcript):
            text, emotions = self.get_emotion(transcript)
            emotion_str = ", ".join(emotions)
            self.emotion_results.append((time_seconds, text, emotion_str))  # Store in global list

        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = self.speech_to_text.client.streaming_recognize(self.speech_to_text.streaming_config,
                                                                   requests)
        self.listen_print_loop(responses, emotion_handler)
