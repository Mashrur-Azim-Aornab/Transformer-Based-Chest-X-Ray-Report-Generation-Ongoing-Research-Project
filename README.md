# Transformer-Based Chest X-Ray Report Generation

An ongoing deep learning project exploring **CNN-based visual tokenization and custom encoder-decoder Transformers for generating radiology findings from chest X-ray images**.

The project is implemented from the ground up in **PyTorch**, with particular emphasis on understanding and experimenting with Transformer components rather than relying entirely on pre-built Transformer implementations.

> **Project status:** Ongoing research/experimental project.
> Model generalization and final report-generation performance are still under evaluation.

---

## Overview

Automatic generation of radiology reports from medical images is a challenging multimodal learning problem. A chest X-ray contains spatially distributed visual information, while a radiology report represents that information as structured natural language.

This project investigates an image-to-text pipeline in which:

```text
Chest X-Ray
    │
    ▼
CNN-based Visual Tokenizer
    │
    ▼
Visual Tokens
    │
    ▼
Transformer Encoder
    │
    ▼
Encoded Visual Representation
    │
    ▼
Transformer Decoder
    │
    ▼
Generated Radiology Findings
```

The current implementation uses a custom CNN-based patch/visual embedding module followed by a Transformer encoder and an autoregressive Transformer decoder.

---

## Dataset

The project uses the **Indiana University Chest X-ray dataset** available through Kaggle.

The dataset contains chest X-ray images together with associated radiology report information. For this project, the **`findings`** field is used as the natural-language generation target.

### Preprocessing

The current pipeline:

* Selects **frontal chest X-ray images**
* Associates images with their corresponding `findings` text
* Removes samples with missing or empty findings
* Converts the report text into token sequences
* Adds special tokens for:

  * `<BOS>` — beginning of sequence
  * `<EOS>` — end of sequence
  * `<PAD>` — padding
  * `<UNK>` — unknown token
* Resizes input images to **224 × 224**
* Uses train/validation/test splits for model development and evaluation

---

## Model Architecture

### 1. CNN-Based Visual Tokenizer

Instead of directly flattening the image into a single vector, the model converts the X-ray into a sequence of spatial visual tokens.

The current tokenizer consists of:

```text
Input X-ray
224 × 224
   │
   ▼
Conv2D
   │
 ReLU
   │
   ▼
Conv2D
   │
 ReLU
   │
   ▼
Projection Conv2D
   │
   ▼
Flatten spatial dimensions
   │
   ▼
Visual token sequence
```

Each visual token has an embedding dimension of **768**.

The project experiments with different kernel/stride configurations to investigate how the number and granularity of visual tokens affect Transformer-based report generation.

### Visual-token experiments

Current experiments include approximately:

| Configuration     | Visual tokens |
| ----------------- | ------------: |
| `patch_size = 12` |           256 |
| `patch_size = 10` |           400 |
| `patch_size = 8`  |           676 |
| `patch_size = 4`  |         2,916 |

The actual number of tokens is measured from the CNN output rather than assumed from the nominal patch size because the preceding convolutional layers change the spatial dimensions.

---

## 2. Transformer Encoder

The visual tokens are processed by a custom Transformer encoder.

Current configuration:

```text
Encoder layers:        2
Attention heads:       8
Input embedding dim:   768
Transformer d_model:   512
```

The implementation includes:

* Multi-head self-attention
* Query/key/value projections
* Positional encoding
* Feed-forward layers
* Residual connections
* Layer normalization
* Dropout

---

## 3. Transformer Decoder

The decoder generates the radiology findings autoregressively from the encoded visual representation.

Current configuration:

```text
Decoder layers:        2
Attention heads:       8
Input embedding dim:   768
Transformer d_model:   512
Maximum sequence length: 149
```

The decoder includes:

* Masked multi-head self-attention
* Encoder-decoder cross-attention
* Feed-forward layers
* Positional encoding
* Residual connections
* Layer normalization
* Output projection to the vocabulary

### Attention masking

The decoder uses a combination of:

1. **Causal masking**
   Prevents the decoder from attending to future tokens.

2. **Padding masking**
   Prevents attention to `<PAD>` positions.

These masks are combined before applying them to the attention scores.

---

## Training

The model is trained using **teacher forcing**.

For a target sequence:

```text
<BOS> a patient ... findings <EOS>
```

the decoder receives the shifted sequence:

```text
<BOS> a patient ... findings
```

and attempts to predict:

```text
a patient ... findings <EOS>
```

The training objective is token-level cross-entropy loss with padding positions ignored.

---

## Experimental Results

The project is currently focused on understanding how visual tokenization affects training and generalization.

One completed experiment using **676 visual tokens** showed the following validation-loss trajectory:

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |        2.6457 |          2.3644 |
|     2 |        2.2857 |          2.1712 |
|     3 |        2.1297 |          2.1242 |
|     4 |        2.0571 |          2.0193 |
|     5 |        1.9701 |          1.9670 |
|     6 |        1.9022 |          1.9527 |
|     7 |        1.8519 |      **1.8869** |

Additional experiments are being conducted with different visual-token resolutions, including **256, 400, 676, and 2,916 tokens**.

The purpose of these experiments is not simply to increase model size, but to investigate whether finer-grained spatial representations improve medical report generation enough to justify the additional computational cost.

> Final test-set performance, caption-generation quality, and generalization are still being evaluated.

---

## Why This Project?

This project was motivated by an interest in the intersection of:

* Computer vision
* Natural language processing
* Medical AI
* Deep learning
* Transformer architectures
* Multimodal learning
* Representation learning
* Generalization under limited/noisy data

A major goal is to understand how information moves from an image representation into a sequence-generation model and how architectural choices affect the resulting system.

---

## Implementation Highlights

The Transformer components are implemented directly in PyTorch, including:

* Multi-head attention
* Self-attention
* Cross-attention
* Positional encoding
* Decoder causal masking
* Padding masking
* Residual connections
* Layer normalization
* Autoregressive sequence generation
* CNN-to-Transformer visual tokenization

The notebook also contains experiments and debugging steps used during development.

---

## Repository Structure

```text
.
├── x_ray to caption transformer model.ipynb
└── README.md
```

> The notebook may have a different filename in the repository. Update the filename above to match the actual file.

---

## Running the Project

The original development environment is **Kaggle with GPU acceleration**.

To reproduce the experiments:

1. Obtain the Indiana University chest X-ray dataset from Kaggle.
2. Update the dataset paths in the notebook if necessary.
3. Run the notebook with a CUDA-enabled GPU.
4. Adjust batch size and model configuration according to available GPU memory.

The larger-token experiments, particularly the **2,916-token configuration**, require substantially more memory because Transformer self-attention scales quadratically with sequence length.

---

## Current Limitations

This is an ongoing research project and several aspects remain under investigation:

* Generalization to unseen X-ray images
* Quality and factual consistency of generated findings
* Comparison of different visual-token resolutions
* Possible overfitting
* Computational cost of high-resolution tokenization
* Evaluation with standard image-captioning/report-generation metrics
* Clinical relevance and factual correctness of generated reports

The system is intended as a research experiment and **not as a clinical diagnostic or reporting system**.

---

## Future Work

Planned directions include:

* More systematic comparison of visual-token resolutions
* Improved evaluation of generated findings
* BLEU, ROUGE, METEOR and related text-generation metrics
* Comparison of generated reports against reference findings
* Investigation of attention patterns and visual-token representations
* More robust evaluation of generalization
* Experimentation with alternative CNN tokenizers
* Exploration of stronger multimodal architectures
* Investigation of domain-shift robustness and limited-data learning

---

## Technologies

```text
Python
PyTorch
TorchVision
NumPy
Pandas
Matplotlib
Kaggle
CUDA
Transformer Architecture
Computer Vision
Natural Language Processing
Medical AI
```

---

## Disclaimer

This repository contains an experimental research implementation for educational and research purposes. Generated text should not be interpreted as medical advice, diagnosis, or a clinically validated radiology report.
