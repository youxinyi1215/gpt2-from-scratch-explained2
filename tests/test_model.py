import torch

from gpt2 import GPT, GPTConfig


def test_forward_shape() -> None:
    config = GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(config)
    idx = torch.randint(0, 20, (4, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 8, 20)
    assert loss is not None


def test_generate_extends_sequence() -> None:
    config = GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(config)
    idx = torch.randint(0, 20, (1, 3))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8)
