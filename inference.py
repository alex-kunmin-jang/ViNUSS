"""
End-to-End Sleep Stage Inference (ViNUSS)

Runs the complete pipeline on raw 224x224 video HDF5 files:
  Stage 1 (PTE + MoViNet-A2, subject-relative micro-motion representation)
  -> Stage 2 (SDM: ShortMamba + bidirectional LongMamba)
without saving intermediate features. Writes <output_path>/predictions.csv
(subject, epoch, prediction, label) and a one-row <output_path>/metrics.csv;
run evaluate.py on predictions.csv for the full metric report.

Usage:
    python inference.py \
        --stage1_ckpt checkpoints/stage1.ckpt \
        --stage2_ckpt checkpoints/stage2.ckpt \
        --data_path /path/to/kvss_h5/224x224 \
        --output_path results/
"""
import os
import time
import argparse
import torch
import yaml
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

from models.movinet import build_movinet
from models.mamba import build_mamba


# KVSS training-set mean / std
MEAN = 0.608
STD = 0.144

# 4-class mapping: Wake(0), Light(1=N1+N2), Deep(2=N3), REM(3=REM)
LABEL_MAP_4CLASS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}


def load_movinet_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained MoViNet model."""
    print(f"[MoViNet] Loading from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    tf_like = checkpoint.get('hyper_parameters', {}).get('model_config', {}).get('pretrained', True)

    model = build_movinet(
        causal=False,
        pretrained=False,
        num_classes=5,
        make_features=True,
        double_loss=False,
        tf_like=tf_like
    )

    state_dict = {(k[len('model.'):] if k.startswith('model.') else k): v
                  for k, v in checkpoint['state_dict'].items()}
    # strict=False only for the ISDL projection head, which is not needed at inference
    result = model.load_state_dict(state_dict, strict=False)
    assert not result.missing_keys, f"Missing keys in Stage 1 checkpoint: {result.missing_keys}"
    assert all(k.startswith('projection_head') for k in result.unexpected_keys), \
        f"Unexpected keys in Stage 1 checkpoint: {result.unexpected_keys}"

    model = model.to(device)
    model.eval()
    print(f"[MoViNet] Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def load_mamba_model(checkpoint_path: str, config_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained Mamba model; model_config is read from the Stage 2 YAML."""
    print(f"[Mamba] Loading from: {checkpoint_path} (model_config: {config_path})")

    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)['model_config']
    model = build_mamba(**model_config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {(k[len('model.'):] if k.startswith('model.') else k): v
                  for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    print(f"[Mamba] Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def normalize_frames(frames: np.ndarray) -> torch.Tensor:
    """Normalize frames: /255 -> z-score."""
    frames = torch.tensor(frames, dtype=torch.float32)
    frames = frames / 255.0
    frames = (frames - MEAN) / STD
    return frames


def extract_features(
    model: torch.nn.Module,
    frames: np.ndarray,
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    """Extract features for all epochs using MoViNet."""
    num_epochs = frames.shape[0]
    all_features = []

    for i in range(0, num_epochs, batch_size):
        end_idx = min(i + batch_size, num_epochs)
        batch_frames = frames[i:end_idx]
        batch_frames = normalize_frames(batch_frames)
        batch_frames = batch_frames.unsqueeze(1)  # (B, 1, T, H, W)
        batch_frames = batch_frames.to(device)

        with torch.no_grad():
            _, features = model(batch_frames)

        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def predict(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device
) -> tuple:
    """Run Mamba inference on extracted features."""
    features = features.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(features)

    logits = logits.squeeze(0).cpu().numpy()
    predictions = logits.argmax(axis=1)

    return predictions, logits


def inference_subject(
    subject_path: str,
    movinet: torch.nn.Module,
    mamba: torch.nn.Module,
    device: torch.device,
    batch_size: int = 4
) -> tuple:
    """Run complete inference pipeline for a single subject."""
    with h5py.File(subject_path, 'r') as f:
        frames = f['frames'][:]
        labels = f['labels'][:]

    features = extract_features(movinet, frames, batch_size, device)
    predictions, logits = predict(mamba, features, device)

    return predictions, labels


def convert_to_4class(labels: np.ndarray) -> np.ndarray:
    """Convert 5-class labels to 4-class (merge N1 and N2)."""
    return np.array([LABEL_MAP_4CLASS[l] for l in labels])


def main():
    parser = argparse.ArgumentParser(description='End-to-End Sleep Stage Inference')
    parser.add_argument('--stage1_ckpt', type=str, required=True, help='Path to Stage 1 (MoViNet) checkpoint')
    parser.add_argument('--stage2_ckpt', type=str, required=True, help='Path to Stage 2 (Mamba) checkpoint')
    parser.add_argument('--stage2_config', type=str, default='configs/stage2_test.yaml', help='Stage 2 YAML (model_config)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to h5 video data directory')
    parser.add_argument('--output_path', type=str, default='./results', help='Output directory')
    parser.add_argument('--split_path', type=str, default='./data/splits', help='Path to split files')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("End-to-End Sleep Stage Inference")
    print("=" * 60)
    print(f"Stage 1 checkpoint: {args.stage1_ckpt}")
    print(f"Stage 2 checkpoint: {args.stage2_ckpt}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {device}")
    print("=" * 60)

    start_time = time.time()
    os.makedirs(args.output_path, exist_ok=True)

    # Load models
    movinet = load_movinet_model(args.stage1_ckpt, device)
    mamba = load_mamba_model(args.stage2_ckpt, args.stage2_config, device)

    # Load test subjects
    test_list_path = os.path.join(args.split_path, 'test.txt')
    with open(test_list_path, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    # Filter available subjects
    available_subjects = [s for s in subjects if os.path.exists(os.path.join(args.data_path, s))]
    missing_subjects = [s for s in subjects if s not in available_subjects]

    print(f"\nTest subjects: {len(subjects)}")
    print(f"Available: {len(available_subjects)}")
    if missing_subjects:
        print(f"[WARNING] {len(missing_subjects)} subject(s) not found in {args.data_path}: {missing_subjects}")

    if len(available_subjects) == 0:
        print("[ERROR] No subjects found!")
        return

    # Run inference
    print("\nRunning inference...")
    all_results = []

    for subject in tqdm(available_subjects, desc="Subjects"):
        h5_path = os.path.join(args.data_path, subject)
        subject_id = subject.replace('.h5', '')

        predictions, labels = inference_subject(
            h5_path, movinet, mamba, device, args.batch_size
        )

        for epoch_idx in range(len(predictions)):
            all_results.append({
                'subject': subject_id,
                'epoch': epoch_idx,
                'prediction': int(predictions[epoch_idx]),
                'label': int(labels[epoch_idx])
            })

    # Save results
    df = pd.DataFrame(all_results)
    output_file = os.path.join(args.output_path, "predictions.csv")
    df.to_csv(output_file, index=False)
    print(f"\nPredictions saved: {output_file}")

    # Calculate metrics
    labels_5c = df['label'].values
    preds_5c = df['prediction'].values
    labels_4c = convert_to_4class(labels_5c)
    preds_4c = convert_to_4class(preds_5c)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Subjects: {len(available_subjects)}")
    print(f"Epochs: {len(df)}")
    print("\n5-Class Metrics:")
    print(f"  Accuracy: {accuracy_score(labels_5c, preds_5c):.4f}")
    print(f"  Macro F1: {f1_score(labels_5c, preds_5c, average='macro'):.4f}")
    print(f"  Kappa:    {cohen_kappa_score(labels_5c, preds_5c):.4f}")
    print("\n4-Class Metrics (N1+N2 merged):")
    print(f"  Accuracy: {accuracy_score(labels_4c, preds_4c):.4f}")
    print(f"  Macro F1: {f1_score(labels_4c, preds_4c, average='macro'):.4f}")
    print(f"  Kappa:    {cohen_kappa_score(labels_4c, preds_4c):.4f}")
    print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print("\n5-Class Classification Report:")
    print(classification_report(labels_5c, preds_5c, digits=3, zero_division=0))

    # Save metrics
    metrics_file = os.path.join(args.output_path, "metrics.csv")
    pd.DataFrame([{
        'num_subjects': len(available_subjects),
        'num_epochs': len(df),
        'accuracy_5class': accuracy_score(labels_5c, preds_5c),
        'macro_f1_5class': f1_score(labels_5c, preds_5c, average='macro'),
        'kappa_5class': cohen_kappa_score(labels_5c, preds_5c),
        'accuracy_4class': accuracy_score(labels_4c, preds_4c),
        'macro_f1_4class': f1_score(labels_4c, preds_4c, average='macro'),
        'kappa_4class': cohen_kappa_score(labels_4c, preds_4c),
        'inference_time_sec': elapsed
    }]).to_csv(metrics_file, index=False)
    print(f"Metrics saved: {metrics_file}")


if __name__ == "__main__":
    main()
