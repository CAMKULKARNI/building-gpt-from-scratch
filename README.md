# Building GPT from Scratch

This repository contains the inner workings of Generative Pre-trained Transformer (GPT) models.

The code is heavily inspired by and follows the structure of Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?si=aOxmRSopPylRL0bJ), [Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=ertj_YE3umZVwgu8) and [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=vCE2p1EXTsaUSgHo) (This is in progress by the way in the GPT-2.py file 😅). These videos have the code explained in [PyTorch](https://pytorch.org/), but I have developed the code in [Tensorflow](https://www.tensorflow.org/) just because I like Tensorflow (please don't hate me for this 🥲). The repository will have changes merged frequently as I learn more about GPT and various LLMs and their nuances like RoPE, their application in Image processing via ViT, etc.

Code cleanup also will be done soon, where the classes will be grouped into one file with their documentations and the singular functions will be grouped into one file with their documentations.

Also, the code is actively written by [Gemini VSCode extension ](https://cloud.google.com/gemini/docs/codeassist/write-code-gemini) (Upto 50% code present in the codebase was suggested by Gemini and 100% code was reviewed by me 😶‍🌫️)

## Core Components

This project is broken down into several key components, each in its own file, to illustrate the concepts progressively.

### 1. Tokenization (`tokenization.ipynb`)

Tokenization is the first crucial step in any language model. This Jupyter notebook provides a deep dive into Byte-Pair Encoding (BPE), a common tokenization strategy.

- **`tokenization.ipynb`**: A hands-on walkthrough of building a BPE tokenizer from scratch. It covers:
  - The difference between character, code point, and byte-level tokenization.
  - The BPE algorithm for iteratively merging frequent byte pairs to build a vocabulary.
  - Implementation of `encode` and `decode` functions.
- **`text.txt`**: A sample text file about Unicode used as the training corpus for our custom tokenizer.

### 2. Bigram Language Model (`BigramModel.py`)

As a baseline, we start with the simplest possible language model: a Bigram model. This model predicts the next character based only on the immediately preceding character.

- **`BigramModel.py`**:
  - Implements a simple lookup-table-based Bigram model using `tf.keras.layers.Embedding`.
  - Includes a full training and generation loop.
  - Serves as a starting point to understand the basic framework of training a language model.

### 3. GPT-like Model (`GPT.py`)

This is the core of the project. We build a decoder-only Transformer model with the key components that make GPTs so powerful.

- **`GPT.py`**:
  - **Self-Attention**: Implementation of a single head of self-attention (`Head` class).
  - **Multi-Head Attention**: Parallel self-attention heads to capture different relational patterns (`MultiHeadAttention` class).
  - **Feed-Forward Network**: A simple MLP that follows the attention block.
  - **Transformer Block**: Combines multi-head attention and feed-forward networks with residual connections and layer normalization.
  - **Full GPT Model**: Stacks multiple transformer blocks to create the final model.
  - Includes training, validation, and text generation logic.

### 4. GPT-2 Implementation (`GPT-2.py`) (In Progress)

This script provides a more faithful implementation of the GPT-2 architecture, demonstrating how the core concepts scale up.

- **`GPT-2.py`**:
  - Follows the GPT-2 architecture with its specific layer configurations.
  - Uses `tiktoken`, OpenAI's fast BPE tokenizer, for tokenization, showing how a production-grade tokenizer is used.
  - Demonstrates text generation with a pre-trained model's architecture.

### Dataset

- **`input.txt`**: The primary training corpus for `BigramModel.py` and `GPT.py`, containing a collection of Shakespeare's works.

## Getting Started

### Prerequisites

Make sure you have Python 3 installed. You can install the necessary libraries using pip:

```bash
pip install tensorflow tqdm
```

For `GPT-2.py`, you will also need `tiktoken`:

```bash
pip install tiktoken
```

The `tokenization-ref.ipynb` notebook requires `regex`:

```bash
pip install regex
```

I use the docker setup to enable Tensorflow with GPU support on my local machine ('AMD Ryzen 5 3550H with Radeon Vega Mobile Gfx' and 'NVIDIA GeForce GTX 1650' 😓)
This is a wonderful video to have it setup [Docker Desktop on Windows 11: WSL Ubuntu 24.04 GPU Integration + NVIDIA Toolkit & VS Code Setup!](https://youtu.be/t7mkHFOeMdA?si=C7tNd3AMwkYr9FzX). To confirm the setup, you can use the code from this [repository](https://github.com/CAMKULKARNI/Tensorflow-GPU) and observe the GPU usage in your Task Manager.

### Usage

1.  **Explore Tokenization**:
    Open and run the cells in `tokenization.ipynb` to understand how BPE tokenization works.

2.  **Train the Bigram Model**:
    This is a good starting point to verify your setup.
    ```bash
    python BigramModel.py
    ```

3.  **Train the GPT Model**:
    This will train the from-scratch Transformer model.
    ```bash
    python GPT.py
    ```
    The script will save the model weights to `gpt_model.weights.h5` and generate a sample text.

4.  **Run the GPT-2 Implementation**:
    This script demonstrates the architecture and uses `tiktoken` for generation.
    ```bash
    python GPT-2.py
    ```

Enjoy building your own GPT!

