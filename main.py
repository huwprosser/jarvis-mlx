import time
import threading
from queue import Queue
from playsound import playsound
from pydub import AudioSegment

from termcolor import colored

from melo.api import TTS
import stt.whisper.transcribe as ts

from stt.VoiceActivityDetection import VADDetector
from llm.phi3 import generate_response
from pydantic import BaseModel


class ChatMLMessage(BaseModel):
    role: str
    content: str


print(
    colored(
        "Welcome to JARVIS-MLX",
        "cyan",
    )
)

print(
    colored(
        "Follow me on X for updates: https://x.com/huwprossercodes",
        "light_grey",
    )
)

tts = TTS(language="EN_NEWEST", device="mps")


class Client:
    def __init__(self, startListening=True, history: list[ChatMLMessage] = []):
        self.listening = False
        self.history = history
        self.vad = VADDetector(self.onSpeechStart, self.onSpeechEnd, sensitivity=0.5)
        self.vad_data = Queue()
        self.tts = tts
        self.stt = ts.FastTranscriber("mlx-community/whisper-large-v3-mlx-4bit")

        if startListening:
            self.toggleListening()
            self.startListening()
            t = threading.Thread(target=self.transcription_loop)
            t.start()

    def startListening(self):
        t = threading.Thread(target=self.vad.startListening)
        t.start()

    def toggleListening(self):
        if not self.listening:
            print()
            print(colored(f"Listening...", "green"))

        while not self.vad_data.empty():
            self.vad_data.get()

        self.listening = not self.listening

    def onSpeechStart(self):
        pass

    def onSpeechEnd(self, data):
        if data.any():
            self.vad_data.put(data)

    def addToHistory(self, content: str, role: str):
        if role == "user":
            print(colored(f"User: {content}", "green"))
        else:
            print(colored(f"Assistant: {content}", "yellow"))

        print()
        if role == "user":
            content = f"""You are a helpful assistant called Jarvis-MLX. Answer the following question in no more than one short sentence. Address me as Sir at all times. Only respond with the dialogue, nothign else.\n\n{content}"""
        self.history.append(ChatMLMessage(content=content, role=role))

    def getHistoryAsString(self):
        final_str = ""
        for message in self.history:
            final_str += f"<|{message.role}|>{message.content}<|end|>\n"

        return final_str

    def transcription_loop(self):
        while True:
            if not self.vad_data.empty():
                data = self.vad_data.get()
                if self.listening and len(data) > 12000:
                    self.toggleListening()
                    transcribed = self.stt.transcribe(data, language="en")
                    self.addToHistory(transcribed["text"], "user")

                    history = self.getHistoryAsString()
                    response = generate_response(history + "\n<|assistant|>")
                    response = (
                        response.split("<|assistant|>")[0].split("<|end|>")[0].strip()
                    )
                    self.addToHistory(response, "assistant")

                    self.speak(response)

    def speak(self, text):
        speaker_ids = self.tts.hps.data.spk2id
        self.tts.tts_to_file(text, speaker_ids["EN-Newest"], "temp.wav", speed=1.0)
        duration = AudioSegment.from_file("temp.wav").duration_seconds
        playsound("temp.wav")

        time.sleep(duration + 1)

        print(duration)
        self.toggleListening()


if __name__ == "__main__":
    jc = Client(startListening=True, history=[])
