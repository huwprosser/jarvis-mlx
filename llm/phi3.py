from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-8bit")


def generate_response(prompt) -> str:
    response = generate(model, tokenizer, prompt=prompt, verbose=False)
    return response
