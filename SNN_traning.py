"""Train a lightweight ANN surrogate and export fixed-point weights for RTL SNN.

Pipeline:
1. Load Bonn classes Z/O/N/S and build 4-channel grouped samples.
2. Normalize to signed 16-bit.
3. Expand 4-channel input to 64 encoder-aligned one-hot features.
4. Train ANN: 64 -> 8 -> 1 binary classifier (numpy implementation).
5. Quantize trained weights to signed 8-bit and export .mem files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from EEG_mem import CLASS_LABEL_MAP, build_4channel_dataset, normalize_to_int16


def expand_to_64_features(samples_int16: np.ndarray) -> np.ndarray:
	"""Map 4 normalized channels into 64 one-hot features (16 bins/channel)."""
	if samples_int16.ndim != 2 or samples_int16.shape[1] != 4:
		raise ValueError(f"Expected [N, 4] input, got shape {samples_int16.shape}")

	x = samples_int16.astype(np.float32) / 32768.0
	bins = np.floor(((x + 1.0) * 0.5) * 16.0).astype(np.int32)
	bins = np.clip(bins, 0, 15)

	n_samples = x.shape[0]
	features = np.zeros((n_samples, 64), dtype=np.float32)
	row_idx = np.arange(n_samples)

	for ch in range(4):
		features[row_idx, (ch * 16) + bins[:, ch]] = 1.0

	return features


def stratified_split(labels: np.ndarray, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	pos_idx = np.where(labels == 1)[0]
	neg_idx = np.where(labels == 0)[0]

	rng.shuffle(pos_idx)
	rng.shuffle(neg_idx)

	pos_cut = int(len(pos_idx) * train_ratio)
	neg_cut = int(len(neg_idx) * train_ratio)

	train_idx = np.concatenate([pos_idx[:pos_cut], neg_idx[:neg_cut]])
	val_idx = np.concatenate([pos_idx[pos_cut:], neg_idx[neg_cut:]])

	rng.shuffle(train_idx)
	rng.shuffle(val_idx)
	return train_idx, val_idx


def balanced_subsample(features: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	if len(labels) <= max_samples:
		return features, labels

	rng = np.random.default_rng(seed)
	pos_idx = np.where(labels == 1)[0]
	neg_idx = np.where(labels == 0)[0]

	half = max_samples // 2
	pos_take = min(len(pos_idx), half)
	neg_take = min(len(neg_idx), max_samples - pos_take)

	pos_choice = rng.choice(pos_idx, size=pos_take, replace=False)
	neg_choice = rng.choice(neg_idx, size=neg_take, replace=False)
	choice = np.concatenate([pos_choice, neg_choice])
	rng.shuffle(choice)

	return features[choice], labels[choice]


def sigmoid(x: np.ndarray) -> np.ndarray:
	clipped = np.clip(x, -30.0, 30.0)
	return 1.0 / (1.0 + np.exp(-clipped))


def evaluate(features: np.ndarray, labels: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> Dict[str, float]:
	hidden_pre = features @ w1
	hidden = np.maximum(hidden_pre, 0.0)
	logits = hidden @ w2
	probs = sigmoid(logits).reshape(-1)

	preds = (probs >= 0.5).astype(np.uint8)
	labels_u8 = labels.astype(np.uint8)

	tp = int(((preds == 1) & (labels_u8 == 1)).sum())
	tn = int(((preds == 0) & (labels_u8 == 0)).sum())
	fp = int(((preds == 1) & (labels_u8 == 0)).sum())
	fn = int(((preds == 0) & (labels_u8 == 1)).sum())

	accuracy = float((tp + tn) / max(len(labels_u8), 1))
	return {
		"accuracy": accuracy,
		"tp": tp,
		"tn": tn,
		"fp": fp,
		"fn": fn,
	}


def train_ann(
	features: np.ndarray,
	labels: np.ndarray,
	seed: int,
	epochs: int,
	batch_size: int,
	learning_rate: float,
	l2: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
	rng = np.random.default_rng(seed)
	n_features = features.shape[1]

	# Hardware-aligned layer sizes: 64 -> 8 -> 1
	w1 = (rng.standard_normal((n_features, 8)) * 0.08).astype(np.float32)
	w2 = (rng.standard_normal((8, 1)) * 0.08).astype(np.float32)

	train_idx, val_idx = stratified_split(labels, train_ratio=0.8, seed=seed)
	x_train, y_train = features[train_idx], labels[train_idx].astype(np.float32)
	x_val, y_val = features[val_idx], labels[val_idx].astype(np.float32)

	pos_count = float((y_train == 1).sum())
	neg_count = float((y_train == 0).sum())
	pos_weight = neg_count / max(pos_count, 1.0)

	for epoch in range(epochs):
		perm = rng.permutation(len(x_train))
		x_train = x_train[perm]
		y_train = y_train[perm]

		for start in range(0, len(x_train), batch_size):
			end = min(start + batch_size, len(x_train))
			xb = x_train[start:end]
			yb = y_train[start:end]

			hidden_pre = xb @ w1
			hidden = np.maximum(hidden_pre, 0.0)
			logits = hidden @ w2
			probs = sigmoid(logits)

			sample_w = np.where(yb.reshape(-1, 1) > 0.5, pos_weight, 1.0).astype(np.float32)
			dlogits = ((probs - yb.reshape(-1, 1)) * sample_w) / max(len(xb), 1)

			grad_w2 = (hidden.T @ dlogits) + (l2 * w2)
			dhidden = dlogits @ w2.T
			dhidden[hidden_pre <= 0.0] = 0.0
			grad_w1 = (xb.T @ dhidden) + (l2 * w1)

			grad_w2 = np.clip(grad_w2, -2.0, 2.0)
			grad_w1 = np.clip(grad_w1, -2.0, 2.0)

			w2 -= learning_rate * grad_w2
			w1 -= learning_rate * grad_w1

		learning_rate *= 0.98

		if (epoch + 1) % 5 == 0 or epoch == 0:
			train_metrics = evaluate(x_train, y_train.astype(np.uint8), w1, w2)
			val_metrics = evaluate(x_val, y_val.astype(np.uint8), w1, w2)
			print(
				f"epoch={epoch + 1:03d} "
				f"train_acc={train_metrics['accuracy']:.4f} "
				f"val_acc={val_metrics['accuracy']:.4f}"
			)

	final_train = evaluate(x_train, y_train.astype(np.uint8), w1, w2)
	final_val = evaluate(x_val, y_val.astype(np.uint8), w1, w2)
	metrics = {
		"train_accuracy": final_train["accuracy"],
		"val_accuracy": final_val["accuracy"],
		"train_tp": final_train["tp"],
		"train_tn": final_train["tn"],
		"train_fp": final_train["fp"],
		"train_fn": final_train["fn"],
		"val_tp": final_val["tp"],
		"val_tn": final_val["tn"],
		"val_fp": final_val["fp"],
		"val_fn": final_val["fn"],
		"positive_weight": float(pos_weight),
	}

	return w1, w2, metrics


def quantize_int8(weights: np.ndarray, target_abs_max: int = 48) -> Tuple[np.ndarray, float]:
	max_abs = float(np.max(np.abs(weights)))
	if max_abs < 1e-9:
		return np.zeros_like(weights, dtype=np.int8), 1.0

	scale = max_abs / float(target_abs_max)
	q = np.round(weights / scale)
	q = np.clip(q, -127, 127).astype(np.int8)
	return q, scale


def int8_to_hex(value: int) -> str:
	return f"{(int(value) & 0xFF):02X}"


def export_weight_files(
	q_w1: np.ndarray,
	q_w2: np.ndarray,
	output_dir: Path,
) -> Dict[str, str]:
	output_dir.mkdir(parents=True, exist_ok=True)

	l1_required = output_dir / "layer1_weights.mem"
	l2_required = output_dir / "layer2_weights.mem"
	l1_compat = output_dir / "layer1_weights_setA.mem"
	l2_compat = output_dir / "layer2_weights_setA.mem"

	l1_flat = []
	for neuron in range(8):
		for inp in range(64):
			l1_flat.append(int(q_w1[inp, neuron]))

	l2_flat = [int(q_w2[inp, 0]) for inp in range(8)]

	if len(l1_flat) != 512:
		raise ValueError(f"Layer1 export size mismatch: {len(l1_flat)} (expected 512)")
	if len(l2_flat) != 8:
		raise ValueError(f"Layer2 export size mismatch: {len(l2_flat)} (expected 8)")

	for path, data in ((l1_required, l1_flat), (l1_compat, l1_flat), (l2_required, l2_flat), (l2_compat, l2_flat)):
		with path.open("w", encoding="ascii") as f:
			for value in data:
				f.write(f"{int8_to_hex(value)}\n")

	return {
		"layer1_required": str(l1_required),
		"layer2_required": str(l2_required),
		"layer1_compat": str(l1_compat),
		"layer2_compat": str(l2_compat),
	}


def parse_args() -> argparse.Namespace:
	root = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(description="Train ANN and export fixed-point weights for Verilog SNN.")
	parser.add_argument("--dataset-root", type=Path, default=root / "EEG-Dataset" / "Dataset")
	parser.add_argument("--output-dir", type=Path, default=root / "Epilepsy_Seizure_Detection_verilog")
	parser.add_argument("--max-groups-per-class", type=int, default=25)
	parser.add_argument("--sample-length", type=int, default=1024)
	parser.add_argument("--max-train-samples", type=int, default=120000)
	parser.add_argument("--seed", type=int, default=2026)
	parser.add_argument("--epochs", type=int, default=40)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--learning-rate", type=float, default=0.06)
	parser.add_argument("--l2", type=float, default=2e-4)
	parser.add_argument("--metrics-json", type=Path, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	class_map: Dict[str, int] = {"Z": CLASS_LABEL_MAP["Z"], "O": CLASS_LABEL_MAP["O"], "N": CLASS_LABEL_MAP["N"], "S": CLASS_LABEL_MAP["S"]}

	raw_samples, labels, group_stats = build_4channel_dataset(
		dataset_root=args.dataset_root,
		class_label_map=class_map,
		max_groups_per_class=args.max_groups_per_class,
		sample_length=args.sample_length,
	)

	samples_int16, raw_min, raw_max = normalize_to_int16(raw_samples)
	features = expand_to_64_features(samples_int16)
	features, labels = balanced_subsample(features, labels, max_samples=args.max_train_samples, seed=args.seed)

	w1, w2, train_metrics = train_ann(
		features=features,
		labels=labels,
		seed=args.seed,
		epochs=args.epochs,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		l2=args.l2,
	)

	q_w1, scale1 = quantize_int8(w1, target_abs_max=48)
	q_w2, scale2 = quantize_int8(w2, target_abs_max=48)

	weight_paths = export_weight_files(q_w1=q_w1, q_w2=q_w2, output_dir=args.output_dir)

	metrics = {
		"dataset_groups_per_class": group_stats,
		"dataset_samples_used": int(len(labels)),
		"raw_min": int(raw_min),
		"raw_max": int(raw_max),
		"q_scale_layer1": float(scale1),
		"q_scale_layer2": float(scale2),
		"layer1_nonzero": int(np.count_nonzero(q_w1)),
		"layer2_nonzero": int(np.count_nonzero(q_w2)),
	}
	metrics.update(train_metrics)
	metrics.update(weight_paths)

	metrics_path = args.metrics_json if args.metrics_json is not None else (args.output_dir / "training_metrics.json")
	metrics_path.parent.mkdir(parents=True, exist_ok=True)
	with metrics_path.open("w", encoding="ascii") as f:
		json.dump(metrics, f, indent=2)

	print("Training and weight export complete")
	print(f"  layer1 weights: {weight_paths['layer1_required']}")
	print(f"  layer2 weights: {weight_paths['layer2_required']}")
	print(f"  train_acc: {metrics['train_accuracy']:.4f}")
	print(f"  val_acc: {metrics['val_accuracy']:.4f}")
	print(f"  metrics: {metrics_path}")


if __name__ == "__main__":
	main()
