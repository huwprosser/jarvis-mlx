import whisper

result = whisper.transcribe(
    "test.mp3", path_or_hf_repo="mlx-community/whisper-large-v3-mlx-4bit"
)
print(result)
