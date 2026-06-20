from pathlib import Path
import random

import torch

from gpt2 import CharTokenizer, GPT, GPTConfig


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def main() -> None:
    torch.manual_seed(7)
    random.seed(7)

    text = Path("data/tiny_sample.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.1,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(150):
        x, y = get_batch(encoded, config.block_size, batch_size=16)
        logits, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "stoi": tokenizer.stoi,
        },
        "checkpoint.pt",
    )
    print("saved checkpoint.pt")


if __name__ == "__main__":
    main()
