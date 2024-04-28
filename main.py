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


print(
    colored(
        """   __    _      __        _____  __                 ____  __
   \ \  /_\    /__\/\   /\\_   \/ _\       /\/\    / /\ \/ /
    \ \//_\\  / \//\ \ / / / /\/\ \ _____ /    \  / /  \  / 
 /\_/ /  _  \/ _  \ \ V /\/ /_  _\ \_____/ /\/\ \/ /___/  \ 
 \___/\_/ \_/\/ \_/  \_/\____/  \__/     \/    \/\____/_/\_\
                                                                                               
""",
        "cyan",
    )
)

print(
    colored(
        "A follow on X would mean the world: https://x.com/huwprossercodes",
        "light_grey",
    )
)
print()

tts = TTS(language="EN_NEWEST", device="mps")


class Client:
    def __init__(self, startListening=True):
        self.listening = False
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

    def transcription_loop(self):
        while True:
            if not self.vad_data.empty():
                data = self.vad_data.get()
                if self.listening and len(data) > 12000:
                    self.toggleListening()
                    transcribed = self.stt.transcribe(data, language="en")
                    print(colored(f"Transcribed: {transcribed['text']}", "green"))
                    response = generate_response(
                        f'<|user|>\n{transcribed["text"]} <|end|>\n<|assistant|>'
                    )
                    response = response.split("<|assistant|>")[0]

                    print(colored(f"Response: {response}", "yellow"))
                    self.speak(response)

    def speak(self, text):
        speaker_ids = self.tts.hps.data.spk2id
        self.tts.tts_to_file(text, speaker_ids["EN-Newest"], "temp.wav", speed=1.0)
        playsound("temp.wav")

        duration = AudioSegment.from_file("temp.wav").duration_seconds
        time.sleep(duration + 1)
        self.toggleListening()


if __name__ == "__main__":
    jc = Client(startListening=True)
