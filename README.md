
# Vision Transformer (ViT) — Paper Replication with PyTorch

This project replicates the model architecture described in the research paper  
**[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)** using PyTorch.

The goal is to understand the structure of the original Vision Transformer and build it from scratch to strengthen our understanding of Transformer-based architectures in computer vision.

---

## 📚 Paper Summary

The Vision Transformer (ViT) adapts the Transformer architecture — originally designed for NLP — to image classification tasks. It does so by:
- Splitting an image into fixed-size patches (e.g., 16x16)
- Flattening and embedding each patch
- Feeding the sequence of patch embeddings into a standard Transformer encoder
- Using a classification token to predict the class of the image

---

## 🔧 Project Structure

```
.
├── vit_model_colab.py     # ViT model for use in Google Colab
├── utils_colab.py         # Utilities for use in Google Colab (training, loading, etc.)
├── vit_model_local.py     # ViT model for local Python environments (e.g., VSCode)
├── utils_local.py         # Utilities for local Python environments
├── README.md              # You're here!
```

---

## 🚀 How to Run

### Option 1: Google Colab

1. Upload `vit_model_colab.py` and `utils_colab.py` to your Colab notebook.
2. In your notebook, use:
   ```python
   from vit_model_colab import ViT
   from utils_colab import *  # your helper functions
   ```

### Option 2: Local Environment (e.g. VSCode, PyCharm)

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vision-transformer-replication.git
   cd vision-transformer-replication
   ```

2. (Optional) Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install torch torchvision
   ```

3. Run an example:
   ```bash
   python main.py
   ```

---

## 🧪 Sample Input & Output

```python
import torch
from vit_model_local import ViT

sample_images = torch.randn(2, 3, 224, 224)
model = ViT(img_size=224, in_channels=3, patch_size=16, num_classes=1000,
            embed_dim=768, num_heads=12, mlp_dim=3072, num_transformer_layers=12)

output = model(sample_images)
print(output.shape)  # torch.Size([2, 1000])
```

---

## ✅ Goals

- 📖 Understand and implement the ViT architecture from scratch
- 🧠 Gain experience in translating research papers into PyTorch code
- ⚙️ Build momentum for replicating future ML papers

---

## 🙌 Credits

- Original paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Inspiration: [Daniel Bourke's](https://github.com/mrdbourke/pytorch-deep-learning) PyTorch tutorials

---
