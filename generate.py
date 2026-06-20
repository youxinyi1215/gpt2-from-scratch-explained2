import argparse

import torch

from gpt2 import GPT, GPTConfig
from gpt2.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="To be")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    checkpoint = torch.load("checkpoint.pt", map_location="cpu")
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = CharTokenizer("")
    tokenizer.stoi = checkpoint["stoi"]
    tokenizer.itos = {idx: ch for ch, idx in tokenizer.stoi.items()}

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=0.8, top_k=20)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
