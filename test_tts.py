from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = "auto"  # Will automatically use GPU if available

# English
model = TTS(language="EN_NEWEST", device=device)
speaker_ids = model.hps.data.spk2id

output_path = "en-newest.wav"

to_say = "x test x"
model.tts_to_file(to_say, speaker_ids["EN-Newest"], output_path, speed=speed)
