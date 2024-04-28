# Jarvis MLX

A all-in-one solution to stark-level productivity running offline on your MacBook using SOTA technology and MLX, Apple's new machine learning framework optimized for Apple Silicon.

Native python is required to run llm libs:
`python -c "import platform; print(platform.processor())"`
should say "arm"

`CONDA_SUBDIR=osx-arm64 conda create -n native numpy -c conda-forge` will get you a osx-arm64

will create a conda env with numpy for arm64 called "native"

## Speech-to-text (STT)

Using Whisper for this. SOTA speech recognition opensourced by openai and trained on 1.5k hours of audio. It's a tiny model that runs on your macbook, upgradable to larger models in the series for better performance.

Helpful link: [Whisper MLX](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
