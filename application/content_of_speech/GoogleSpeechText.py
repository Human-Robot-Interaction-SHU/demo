from google.cloud import speech
from google.oauth2 import service_account


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


class GoogleSpeechToText:
    def __init__(self):
        self.language_code = "en-US"
        self.cred = service_account.Credentials.from_service_account_info(google_config)
        self.client = speech.SpeechClient(credentials=self.cred)
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self.language_code,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config, interim_results=True
        )
