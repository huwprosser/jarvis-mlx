# Jarvis MLX

⚠️ Work in progress! [Follow me on X for updates!](https://x.com/huwprossercodes)

An all-in-one solution to stark-level productivity running offline on your MacBook using SOTA technology and MLX, Apple's new machine learning framework optimized for Apple Silicon.

Native python is required to run MLX libs:
`python -c "import platform; print(platform.processor())"`
should say "arm"

`CONDA_SUBDIR=osx-arm64 conda create -n native numpy -c conda-forge`
will create a conda env with numpy for arm64 called "native"

Firstly, pip install the requirements:
`pip install -r requirements.txt`

## Speech-to-text (STT)

Using Whisper for this. SOTA speech recognition opensourced by openai and trained on 1.5k hours of audio. It's a tiny model that runs on your macbook, upgradable to larger models in the series for better performance.

Helpful link: [Whisper MLX](https://github.com/ml-explore/mlx-examples/tree/main/whisper)

## Large Language Model

Using Phi 3 out the box you can achieve 60 tokens per second on an M1 Max. You can also finetune your own models and load them in. I'd highly recommend Mistral or Llama 3.

## Text-to-speech (TTS)

For this, I opted to use [MeloTTS](https://github.com/myshell-ai/MeloTTS?tab=readme-ov-file). It's not as hyped up as some other offerings but it's fast, runs on a mac and finetunable on custom data using the original repo. A stripped down version of the inference code for english can be found in [/melo](melo/).

###### Please note, out the box Jarvis-mlx has a female voice. You will need to train your own MeloTTS model to change this. Finetuning Phi 3 will also achieve much better behaviour.

PRs Welcome!
