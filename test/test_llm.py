import pytest
from mlx_lm import load, generate


def test_llm():
    model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-8bit")
    response = generate(model, tokenizer, prompt="hello", verbose=True)
    assert len(response) > 0
