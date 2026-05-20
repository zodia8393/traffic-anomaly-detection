"""Level 3 — LSTM AutoEncoder 기반 시계열 이상 탐지.

정상 주행 시퀀스의 reconstruction을 학습하고,
reconstruction error가 높으면 이상으로 판정.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

from .aihub_loader import load_clips, Clip

# ── 시퀀스 특성 추출 ────────────────────────────────────────────────

SEQ_FEATURES = [
    "cx", "cy", "w", "h",
    "dx", "dy", "speed",
    "heading_sin", "heading_cos",
    "lateral_speed", "area_change_rate", "aspect_ratio",
]
N_FEAT = len(SEQ_FEATURES)
SEQ_LEN = 20


def clip_to_sequences(clip: Clip) -> list[tuple[np.ndarray, int]]:
    """클립 → (SEQ_LEN, N_FEAT) 시퀀스 리스트 반환."""
    from collections import defaultdict
    import math

    tracks: dict[int, list[tuple[int, list]]] = defaultdict(list)
    for frame in clip.frames:
        for v in frame.vehicles:
            vid = v.get("id", 0)
            bbox = v.get("bbox", [0, 0, 0, 0])
            tracks[vid].append((frame.frame_idx, bbox))

    label = 0 if clip.is_normal else 1
    sequences = []

    for vid, seq in tracks.items():
        if len(seq) < SEQ_LEN:
            continue

        raw = []
        for i, (fidx, bbox) in enumerate(seq):
            x, y, w, h = bbox
            cx, cy = x + w / 2, y + h / 2
            ar = w / h if h > 0 else 1.0

            if i > 0:
                prev_bbox = seq[i - 1][1]
                px = prev_bbox[0] + prev_bbox[2] / 2
                py = prev_bbox[1] + prev_bbox[3] / 2
                dx, dy = cx - px, cy - py
                speed = math.hypot(dx, dy)
                heading_rad = math.atan2(dx, -dy)
                h_sin, h_cos = math.sin(heading_rad), math.cos(heading_rad)

                # lateral speed (진행방향 수직 성분)
                fwd_x, fwd_y = math.sin(heading_rad), -math.cos(heading_rad)
                lon = dx * fwd_x + dy * fwd_y
                lat = math.hypot(dx - lon * fwd_x, dy - lon * fwd_y)

                # area change rate
                prev_area = prev_bbox[2] * prev_bbox[3]
                curr_area = w * h
                acr = (curr_area - prev_area) / max(prev_area, 1) if prev_area > 0 else 0
            else:
                dx, dy, speed, h_sin, h_cos, lat, acr = 0, 0, 0, 0, 1, 0, 0

            raw.append([
                cx / 1280, cy / 720, w / 1280, h / 720,
                dx / 50, dy / 50, speed / 50,
                h_sin, h_cos,
                lat / 20, acr, ar,
            ])

        raw = np.array(raw, dtype=np.float32)

        for start in range(0, len(raw) - SEQ_LEN + 1, SEQ_LEN // 2):
            window = raw[start:start + SEQ_LEN]
            if len(window) == SEQ_LEN:
                sequences.append((window, label))

    return sequences


class SeqDataset(Dataset):
    def __init__(self, sequences: list[tuple[np.ndarray, int]]):
        self.X = torch.tensor(np.array([s[0] for s in sequences]), dtype=torch.float32)
        self.y = torch.tensor(np.array([s[1] for s in sequences]), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── LSTM AutoEncoder ────────────────────────────────────────────────

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim: int = N_FEAT, hidden_dim: int = 32, latent_dim: int = 16, n_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, n_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        enc_out, (h, c) = self.encoder(x)
        latent = self.enc_fc(h[-1])  # (batch, latent_dim)

        dec_input = self.dec_fc(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_input)
        return dec_out


def compute_reconstruction_error(model: LSTMAutoEncoder, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        recon = model(X)
        errors = ((X - recon) ** 2).mean(dim=(1, 2)).numpy()
    return errors


# ── 학습 + 평가 ────────────────────────────────────────────────────

def train_and_evaluate(
    label_root: Path,
    model_dir: Path,
    max_clips: int | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    t0 = time.time()

    print("클립 로딩...")
    clips = load_clips(label_root, max_clips=max_clips)
    print(f"  {len(clips)}개 클립")

    print("시퀀스 추출...")
    normal_seqs, abnormal_seqs = [], []
    for clip in clips:
        seqs = clip_to_sequences(clip)
        for s in seqs:
            if s[1] == 0:
                normal_seqs.append(s)
            else:
                abnormal_seqs.append(s)

    print(f"  정상 시퀀스: {len(normal_seqs)}, 비정상: {len(abnormal_seqs)}")

    if not normal_seqs:
        print("정상 시퀀스 없음")
        return

    # 학습/검증 분할 (정상의 80% 학습, 20% + 비정상 평가)
    np.random.seed(42)
    idx = np.random.permutation(len(normal_seqs))
    split = int(len(idx) * 0.8)
    train_seqs = [normal_seqs[i] for i in idx[:split]]
    val_normal = [normal_seqs[i] for i in idx[split:]]

    train_ds = SeqDataset(train_seqs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LSTMAutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\n학습 시작 (정상 {len(train_seqs)}개, {epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, _ in train_loader:
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        avg_loss = total_loss / len(train_ds)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.6f}")

    # 평가
    print("\n평가...")
    eval_seqs = val_normal + abnormal_seqs
    eval_ds = SeqDataset(eval_seqs)
    eval_X = eval_ds.X
    eval_y = eval_ds.y.numpy()

    errors = compute_reconstruction_error(model, eval_X)

    # threshold: 정상 검증셋의 95 percentile
    normal_errors = errors[:len(val_normal)]
    threshold = np.percentile(normal_errors, 95)
    print(f"  threshold (정상 p95): {threshold:.6f}")

    y_pred = (errors > threshold).astype(int)

    print(f"\n=== Level 3 LSTM-AE 평가 ===")
    print(classification_report(
        eval_y, y_pred,
        target_names=["정상", "비정상"],
        digits=3,
    ))

    if len(np.unique(eval_y)) > 1:
        auc = roc_auc_score(eval_y, errors)
        print(f"ROC-AUC: {auc:.3f}")

    print(f"\n정상 recon error: mean={normal_errors.mean():.6f}, std={normal_errors.std():.6f}")
    abnormal_errors = errors[len(val_normal):]
    print(f"비정상 recon error: mean={abnormal_errors.mean():.6f}, std={abnormal_errors.std():.6f}")
    print(f"분리도 (비정상mean/정상mean): {abnormal_errors.mean()/max(normal_errors.mean(), 1e-9):.2f}x")

    # 저장
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "threshold": threshold,
        "normal_mean": float(normal_errors.mean()),
        "normal_std": float(normal_errors.std()),
    }, model_dir / "lstm_ae_l3.pt")
    print(f"\n모델 저장: {model_dir / 'lstm_ae_l3.pt'}")
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    label_root = Path("/DATA/aihub_71566/labels/val_extracted")
    model_dir = Path("/workspace/CCTV차종분류/사고분석_설계/models")
    train_and_evaluate(label_root, model_dir)
