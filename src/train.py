import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .model_baseline import build_model
from .data import make_loaders


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

        if i % 50 == 0:
            print(f"  batch {i}/{len(loader)}")

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def main(args):
    # device selection (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # dataloaders 
    train_ld, val_ld, classes = make_loaders(
        data_dir=args.data,
        img_size=args.img_size,
        batch_size=args.batch,
        num_workers=args.workers,
    )
    print(f"classes: {len(classes)}")

    # model
    model = build_model(len(classes)).to(device)

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.optim == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=1e-4)
    elif args.optim == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=1e-4)
    else:  # adamw
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          weight_decay=1e-4)

    # LR scheduler 
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch:03d}")
        train_loss, train_acc = train_one_epoch(
            model, train_ld, optimizer, device, criterion
        )
        val_loss, val_acc = evaluate(model, val_ld, device, criterion)
        scheduler.step()

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "img_size": args.img_size,
                },
                ckpt_dir / "best_top1.pth",
            )

    print(f"\nbest val acc: {best_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
        help="optimizer to use",
    )
    args = parser.parse_args()
    main(args)
