# GPT-2 From Scratch Explained

This repository implements a compact GPT-2 style language model in PyTorch and explains the main parts of the architecture.

The goal is educational clarity, not training a full 1.5B parameter GPT-2 model.

## What Is Implemented

- Token embedding
- Positional embedding
- Causal self-attention
- Multi-head attention
- Feed-forward network
- Transformer block with residual connections and layer normalization
- Autoregressive next-token training
- Text generation with temperature and top-k sampling

## Project Structure

```text
gpt2-from-scratch-explained/
├── README.md
├── requirements.txt
├── data/
│   └── tiny_sample.txt
├── gpt2/
│   ├── __init__.py
│   ├── model.py
│   └── tokenizer.py
├── generate.py
├── train.py
└── tests/
    └── test_model.py
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python generate.py --prompt "To be"
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python generate.py --prompt "To be"
```

## GPT-2 Architecture Explanation

GPT-2 is a decoder-only Transformer language model. It receives a sequence of token IDs and predicts the next token at each position.

### 1. Token Embedding

Each token ID is mapped to a vector:

```text
token_id -> token_embedding
```

### 2. Positional Embedding

Transformers do not naturally know token order, so GPT-2 adds a learned position vector to every token vector:

```text
input = token_embedding + position_embedding
```

### 3. Causal Self-Attention

Self-attention lets each token look at previous tokens. GPT-2 uses a causal mask so position `t` cannot see future positions.

```text
attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

### 4. Multi-Head Attention

The model runs several attention heads in parallel. Each head can learn a different relationship between tokens.

### 5. Feed-Forward Network

After attention, each position passes through a small MLP:

```text
Linear -> GELU -> Linear
```

### 6. Residual Connections and LayerNorm

Residual connections stabilize training by adding the input back to each block output.

```text
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

### 7. Language Modeling Loss

The model predicts the next token:

```text
input:  "To be or"
target: "be or not"
```

Training uses cross-entropy loss over the shifted target sequence.

## Training

The included `train.py` trains a tiny GPT-2 style model on `data/tiny_sample.txt`.

```bash
python train.py
```

The script saves a checkpoint:

```text
checkpoint.pt
```

## Generation

```bash
python generate.py --prompt "To be" --max-new-tokens 80
```

## Tests

```bash
pytest
```

## Notes

The original GPT-2 has many more parameters and was trained on a very large dataset. This repository keeps the same core idea but uses a small configuration so it can be read, tested, and trained on a normal computer.
