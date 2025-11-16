# ATminiCNN

Hello! This is a lightweight convolutional neural network (CNN) project built for experimentation and quick training.

This repository includes:
- A tiny custom CNN implemented from scratch (`model_baseline.py`)
- A configurable training loop (`train.py`)
- A flexible data loader that supports:
    - Tiny CIFAR-10 subset (200 training images / 50 validation images)
    - CIFAR-10
    - CIFAR-100
    - PKL-formatted datasets (original assignment format)

---

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/annetteptran/ATminiCNN
    cd ATminiCNN
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train model:
    ```bash
    python -m src.train
    ```

(Default optimizer = AdamW, lr = 0.001, epochs = 5)

---

### Arguments

| Argument       | Description                             | Example               |
|----------------|-----------------------------------------|-----------------------|
| `--epochs`     | number of training epochs               | `--epochs 10`         |
| `--dataset`    | dataset to use (`tiny_cifar`, `cifar10`, `cifar100`) | `--dataset cifar10`   |
| `--batch-size` | batch size for training                 | `--batch-size 128`    |
| `--lr`         | learning rate                           | `--lr 0.001`          |

---

## Results!

Training was run for **10 epochs** on the **Tiny CIFAR subset**, a reduced dataset containing only **200 training images** and **50 validation images**. Because the dataset is extremely small and the model is intentionally lightweight, accuracy is expected to be modest.

### Final Performance
- **Best Validation Accuracy:** **34.5%**
- **Final Training Accuracy:** ~28%
- **Training Duration:** ~10 epochs

---

## Training Log
```
epoch 001 | train loss 2.259 acc 0.173 | val loss 2.200 acc 0.208

epoch 002
  batch 0/16
epoch 002 | train loss 2.091 acc 0.204 | val loss 2.850 acc 0.192

epoch 003
  batch 0/16
epoch 003 | train loss 2.040 acc 0.225 | val loss 1.984 acc 0.245

epoch 004
  batch 0/16
epoch 004 | train loss 1.988 acc 0.251 | val loss 1.879 acc 0.269

epoch 005
  batch 0/16
epoch 005 | train loss 1.965 acc 0.254 | val loss 1.847 acc 0.309

epoch 006
  batch 0/16
epoch 006 | train loss 1.922 acc 0.278 | val loss 1.786 acc 0.316

epoch 007
  batch 0/16
epoch 007 | train loss 1.917 acc 0.277 | val loss 1.761 acc 0.324

epoch 008
  batch 0/16
epoch 008 | train loss 1.876 acc 0.280 | val loss 1.759 acc 0.324

epoch 009
  batch 0/16
epoch 009 | train loss 1.879 acc 0.287 | val loss 1.715 acc 0.345

epoch 010
  batch 0/16
epoch 010 | train loss 1.871 acc 0.283 | val loss 1.709 acc 0.338

best val acc: 0.345
```
