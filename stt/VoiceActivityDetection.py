import time
import wave
import pyaudio
import webrtcvad
import contextlib
import collections
import numpy as np
import sounddevice as sd

RATE = 16000
CHUNK = 160
CHANNELS = 1
FORMAT = pyaudio.paInt16

audio = pyaudio.PyAudio()


class VADDetector:
    def __init__(self, onSpeechStart, onSpeechEnd, sensitivity=0.4):
        self.channels = [1]
        self.mapping = [c - 1 for c in self.channels]
        self.device_info = sd.query_devices(None, "input")
        self.sample_rate = 16000  # int(self.device_info['default_samplerate'])
        self.interval_size = 10  # audio interval size in ms
        self.sensitivity = sensitivity  # Seconds
        self.block_size = self.sample_rate * self.interval_size / 1000
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)
        self.frameHistory = [False]
        self.block_since_last_spoke = 0
        self.onSpeechStart = onSpeechStart
        self.onSpeechEnd = onSpeechEnd
        self.voiced_frames = collections.deque(maxlen=1000)

    def write_wave(self, path, audio, sample_rate):
        with contextlib.closing(wave.open(path, "w")) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframesraw(audio)

    def voice_activity_detection(self, audio_data):
        return self.vad.is_speech(audio_data, self.sample_rate)

    def audio_callback(self, indata, frames, time, status):
        audio_data = indata
        detection = self.voice_activity_detection(audio_data)

        if self.frameHistory[-1] == True and detection == True:
            self.onSpeechStart()
            self.voiced_frames.append(audio_data)
            self.block_since_last_spoke = 0
        else:
            if (
                self.block_since_last_spoke
                == self.sensitivity * 10 * self.interval_size
            ):

                if len(self.voiced_frames) > 0:
                    samp = b"".join(self.voiced_frames)
                    self.onSpeechEnd(np.frombuffer(samp, dtype=np.int16))
                self.voiced_frames = []
            else:
                # if last block was not speech don't add
                if len(self.voiced_frames) > 0:
                    self.voiced_frames.append(audio_data)

            self.block_since_last_spoke += 1

        self.frameHistory.append(detection)

    def startListening(self):
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_callback(data, CHUNK, time.time(), None)
            except Exception as e:
                print(e)
                break


if __name__ == "__main__":

    def onSpeechStart():
        print("Speech started")

    def onSpeechEnd(data):
        print("Speech ended")
        print(f"Data {data}")

    vad = VADDetector(onSpeechStart, onSpeechEnd)
    vad.startListening()
