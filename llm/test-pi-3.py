from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-8bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)
