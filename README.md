Hello! This is lightweight convolutional neural network (CNN) project built for experimentation and quick training!

This repository includes:
- A tiny custom CNN implemented from scratch (model_baseline.py)
- A configurable training loop (train.py)
- A flexible data loader that supports:
    - Tiny CIFAR-10 subset (2,000 training images / 1,000 validation images)
    - CIFAR-100
    - PKL-formatted datasets (original assignment format)

How to run:
1. conda activate tinycnn
2. python -m src.train
(Default optimizer = AdamW, lr = 0.001, epochs = 5)

Argument	Description	                      Example
--epochs	number of training epochs	      --epochs 10
--lr	    learning rate	                  --lr 0.001
--data	    dataset folder (default: data/)	  --data myfolder/