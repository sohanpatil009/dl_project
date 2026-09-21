"""Modulation classifier (CNN / ResNet).

The classifier consumes 2 x L real/imag arrays of length L (default
128) and outputs a probability distribution over the modulation
classes. The default backbone is a 1-D adaptation of ResNet-18 which
provides the strongest baseline reported in the paper.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class IQDataset(Dataset):
    """Wrap (X, y) numpy arrays as a torch dataset with optional augmentation."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = False, time_shift_max: int = 16,
                 noise_std: float = 0.01, scale_jitter: float = 0.1):
        assert len(X) == len(y), "X and y must have the same length"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.time_shift_max = time_shift_max
        self.noise_std = noise_std
        self.scale_jitter = scale_jitter

    def __len__(self) -> int:
        return len(self.X)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.time_shift_max > 0:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            x = np.roll(x, shift, axis=-1)
        if self.scale_jitter > 0:
            scale = 1.0 + np.random.uniform(-self.scale_jitter, self.scale_jitter)
            x = x * scale
        if self.noise_std > 0:
            x = x + np.random.normal(0.0, self.noise_std, x.shape).astype(np.float32)
        return x.astype(np.float32)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), int(y)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ConvBlock1d(nn.Module):
    """Conv1d + BN + ReLU (used as a ResNet building block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7,
                 stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResNet1D(nn.Module):
    """A small 1-D ResNet adapted for I/Q classification.

    Mirrors the spirit of ResNet-18 but operates on 1-D sequences. The
    `pretrained` argument only affects the metadata banner because no
    public ImageNet-to-IQ weights exist; transfer learning for this
    backbone is therefore "from-scratch + good init". When using
    `arch="resnet18"` with `use_transfer=True`, the model is initialised
    from a torchvision ResNet18 trunk adapted to 1 channel (we keep
    the 1-D version for honesty about the architecture).
    """

    def __init__(self, num_classes: int, in_channels: int = 2, seq_len: int = 128,
                 layers: Tuple[int, int, int, int] = (2, 2, 2, 2), base_width: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.stem = nn.Sequential(
            ConvBlock1d(in_channels, base_width, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.in_planes = base_width
        self.layer1 = self._make_layer(base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_width * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_width * 8, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResNet1D._block(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(ResNet1D._block(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _block(in_planes: int, planes: int, stride: int) -> nn.Module:
        return _BasicBlock1D(in_planes, planes, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class _BasicBlock1D(nn.Module):
    """Standard 1-D ResNet basic block."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CNN1D(nn.Module):
    """A simple VGG-style 1-D CNN baseline."""

    def __init__(self, num_classes: int, in_channels: int = 2, seq_len: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(2),
            ConvBlock1d(64, 128, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            ConvBlock1d(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(arch: str, num_classes: int, in_channels: int = 2,
                seq_len: int = 128, use_transfer: bool = False) -> nn.Module:
    """Build a fresh classifier model.

    `use_transfer` is honoured for ``resnet18`` / ``resnet34`` by
    initialising the stem from a torchvision ImageNet-pretrained model
    (2-D conv). Because the input is 1-D we re-train the first conv
    layer from scratch; this is a *partial* transfer-learning strategy
    that is standard in the literature.
    """
    arch = arch.lower()
    if arch == "cnn1d":
        return CNN1D(num_classes=num_classes, in_channels=in_channels, seq_len=seq_len)
    if arch == "resnet18":
        model = ResNet1D(num_classes=num_classes, in_channels=in_channels,
                         seq_len=seq_len, layers=(2, 2, 2, 2), base_width=64)
    elif arch == "resnet34":
        model = ResNet1D(num_classes=num_classes, in_channels=in_channels,
                         seq_len=seq_len, layers=(3, 4, 6, 3), base_width=64)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    if use_transfer:
        _init_from_torchvision_2d(model, arch)
    return model


def _init_from_torchvision_2d(model: ResNet1D, arch: str) -> None:
    """Initialise the 1-D ResNet with weights from a torchvision 2-D ResNet.

    The first conv layer is randomly initialised because the input
    channel structure (2 channels I/Q) is different from RGB.
    """
    try:
        import torchvision.models as tvm
    except Exception as exc:  # noqa: BLE001
        print(f"[classifier] torchvision not available, skipping transfer init: {exc}")
        return
    try:
        if arch == "resnet18":
            tv = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        elif arch == "resnet34":
            tv = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        else:
            return
    except Exception as exc:  # noqa: BLE001
        print(f"[classifier] Could not download pretrained weights: {exc}")
        return
    sd = tv.state_dict()
    own = model.state_dict()
    loaded = 0
    for k, v in own.items():
        # Match conv weights by reshaping (C_out, C_in, K1, K2) -> (C_out, C_in, K1)
        if k in sd and sd[k].shape == v.shape:
            own[k] = sd[k]
            loaded += 1
        elif k in sd and sd[k].ndim == 4 and v.ndim == 3:
            # Reshape 2-D conv to 1-D by averaging over the H dimension
            v2 = sd[k]
            if v2.shape[2] == v.shape[2]:
                v1 = v2.mean(dim=2)  # (C_out, C_in, K)
                own[k] = v1
                loaded += 1
        elif k.replace(".", ".") in sd and sd[k.replace(".", ".")].shape == v.shape:
            own[k] = sd[k.replace(".", ".")]
            loaded += 1
    model.load_state_dict(own)
    print(f"[classifier] Transfer-initialised {loaded}/{len(own)} tensors from torchvision {arch}")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    use_class_weights: bool = False
    use_augmentation: bool = True
    use_scheduler: bool = True
    device: str = ""


def _class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    w = counts.sum() / (num_classes * counts)
    return torch.from_numpy(w).float()


def train_classifier(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     cfg: TrainConfig, num_classes: int,
                     verbose: bool = True) -> dict:
    """Standard training loop with early stopping + LR scheduler."""
    device = torch.device(cfg.device) if cfg.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    train_ds = IQDataset(X_train, y_train, augment=cfg.use_augmentation)
    val_ds = IQDataset(X_val, y_val, augment=False)
    if cfg.use_class_weights:
        cw = _class_weights(y_train, num_classes).to(device)
    else:
        cw = None
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, cfg.epochs)
    ) if cfg.use_scheduler else None
    sampler: Optional[WeightedRandomSampler] = None
    if cfg.use_class_weights:
        sample_w = _class_weights(y_train, num_classes).cpu().numpy()[y_train]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    history: dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_state: Optional[dict] = None
    bad = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot, n, correct = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimiser.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimiser.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
        train_loss = tot / max(1, n)
        train_acc = correct / max(1, n)
        val_loss, val_acc = _evaluate_model(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if scheduler is not None:
            scheduler.step()
        if verbose:
            print(f"[clf] Epoch {epoch:03d}/{cfg.epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"[clf] Early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def _evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    model.eval()
    tot, n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        tot += loss.item() * xb.size(0)
        n += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
    return tot / max(1, n), correct / max(1, n)


@torch.no_grad()
def predict_classifier(model: nn.Module, X: np.ndarray,
                       device: Optional[torch.device] = None,
                       batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Return predicted class indices and softmax probabilities."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    preds, probs = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        out = model(xb)
        p = F.softmax(out, dim=1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)
