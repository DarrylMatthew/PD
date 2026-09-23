import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = {"healthy": 0, "parkinson": 1}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15, fill=255),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(data_root, dataset_name):
    """Reads data_root/dataset_name/{healthy,parkinson}/<subjectID>_<n>.png"""
    samples = []
    for cls_name, label in CLASSES.items():
        folder = Path(data_root) / dataset_name / cls_name
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                subject = f"{dataset_name}_{cls_name}_{p.stem.split('_')[0]}"
                samples.append((str(p), label, subject))
    if not samples:
        raise FileNotFoundError(f"No images found for {dataset_name} in {data_root}")
    return samples


def split_by_subject(samples, n_splits, seed):
    labels = [s[1] for s in samples]
    groups = [s[2] for s in samples]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, te_idx = next(sgkf.split(np.zeros(len(samples)), labels, groups))
    return [samples[i] for i in tr_idx], [samples[i] for i in te_idx]


class SpiralDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, _ = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, probs, ys = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            probs.append(torch.softmax(logits, 1)[:, 1].detach().cpu())
            ys.append(y.cpu())
    probs = torch.cat(probs).numpy()
    ys = torch.cat(ys).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": total_loss / len(ys),
        "acc": accuracy_score(ys, preds),
        "f1": f1_score(ys, preds, zero_division=0),
        "auc": roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan"),
    }


def report(name, m):
    print(f"{name:30s} acc {m['acc']:.3f} | f1 {m['f1']:.3f} | auc {m['auc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--train-dataset", required=True)
    ap.add_argument("--test-datasets", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    samples = collect_samples(args.data_root, args.train_dataset)
    trainval, test = split_by_subject(samples, 5, args.seed)  # ~20% subjects for test
    train, val = split_by_subject(trainval, 5, args.seed)     # ~20% of the rest for val
    print(f"Images: train {len(train)}, val {len(val)}, test {len(test)}")

    def make_loader(s, tf, shuffle):
        return DataLoader(SpiralDataset(s, tf), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)

    train_dl = make_loader(train, train_tf, True)
    val_dl = make_loader(val, eval_tf, False)
    test_dl = make_loader(test, eval_tf, False)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ckpt = f"best_{args.train_dataset}.pt"
    best_f1 = -1.0

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, device, optimizer)
        va = run_epoch(model, val_dl, device)
        print(f"Epoch {ep:02d} | train loss {tr['loss']:.3f} acc {tr['acc']:.3f} "
              f"| val acc {va['acc']:.3f} f1 {va['f1']:.3f}")
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    print("\n== Results ==")
    report(f"in-domain ({args.train_dataset})", run_epoch(model, test_dl, device))
    for name in args.test_datasets:
        cross = collect_samples(args.data_root, name)
        report(f"cross-dataset ({name})", run_epoch(model, make_loader(cross, eval_tf, False), device))


if __name__ == "__main__":
    main()