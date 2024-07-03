from google.cloud import speech
from google.oauth2 import service_account

class StreamingTranscriber:
    def __init__(self, service_account_info, language_code='en-US', sample_rate_hertz=48000, encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16):
        self.credentials = service_account.Credentials.from_service_account_info(service_account_info)
        self.client = speech.SpeechClient(credentials=self.credentials)
        self.language_code = language_code
        self.sample_rate_hertz = sample_rate_hertz
        self.encoding = encoding

    def get_config(self):
        return speech.RecognitionConfig(
            encoding=self.encoding,
            sample_rate_hertz=self.sample_rate_hertz,
            language_code=self.language_code
        )

    def get_streaming_config(self):
        return speech.StreamingRecognitionConfig(
            config=self.get_config(),
            interim_results=False
        )

    async def transcribe_streaming(self, audio_generator):
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
        responses = self.client.streaming_recognize(config=self.get_streaming_config(), requests=requests)

        for response in responses:
            for result in response.results:
                yield result.alternatives[0].transcript

