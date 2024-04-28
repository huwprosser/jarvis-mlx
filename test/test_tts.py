import os
import pytest
from melo.api import TTS


def test_tts():
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

    # Check if the file was created
    assert os.path.isfile(output_path)

    # Clean up after the test
    os.remove(output_path)
