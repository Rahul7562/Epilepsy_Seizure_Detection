"""Generate Verilog-ready EEG dataset memory files from Bonn EEG folders.

This script loads the required Bonn classes (Z, O, N, S), builds 4-channel
samples by combining four different files, normalizes amplitudes to signed
16-bit values, and exports:

* eeg_dataset.mem  -> HEX values, 4 values per line (one sample)
* labels.mem       -> binary labels per line (0 normal, 1 seizure)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


CLASS_LABEL_MAP: Dict[str, int] = {
	"Z": 0,
	"O": 0,
	"N": 0,
	"S": 1,
}


def _collect_class_files(dataset_root: Path, class_name: str) -> List[Path]:
	class_dir = dataset_root / class_name
	if not class_dir.exists():
		raise FileNotFoundError(f"Missing class directory: {class_dir}")

	files = sorted(
		[p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
		key=lambda p: p.name.lower(),
	)
	if not files:
		raise FileNotFoundError(f"No .txt files found in {class_dir}")
	return files


def _load_signal(file_path: Path, sample_length: Optional[int]) -> np.ndarray:
	signal = np.loadtxt(file_path, dtype=np.int32)
	if signal.ndim != 1:
		signal = signal.reshape(-1)
	if sample_length is not None:
		if len(signal) < sample_length:
			raise ValueError(
				f"File {file_path} has {len(signal)} samples, expected at least {sample_length}."
			)
		signal = signal[:sample_length]
	return signal


def build_4channel_dataset(
	dataset_root: Path,
	class_label_map: Dict[str, int],
	max_groups_per_class: Optional[int] = None,
	sample_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
	"""Build dataset by combining 4 different files into one 4-channel stream."""
	samples_list: List[np.ndarray] = []
	labels_list: List[np.ndarray] = []
	group_stats: Dict[str, int] = {}
	grouped_samples: Dict[str, List[np.ndarray]] = {}
	grouped_labels: Dict[str, List[np.ndarray]] = {}

	for class_name, label in class_label_map.items():
		class_files = _collect_class_files(dataset_root, class_name)
		groups_available = len(class_files) // 4
		if groups_available == 0:
			raise ValueError(f"Class {class_name} does not have enough files for 4-channel grouping.")

		groups_to_use = groups_available
		if max_groups_per_class is not None:
			groups_to_use = min(groups_to_use, max_groups_per_class)

		group_stats[class_name] = groups_to_use
		grouped_samples[class_name] = []
		grouped_labels[class_name] = []

		for group_idx in range(groups_to_use):
			group_files = class_files[group_idx * 4 : (group_idx + 1) * 4]
			channel_signals = [_load_signal(fp, sample_length=sample_length) for fp in group_files]

			local_len = min(len(sig) for sig in channel_signals)
			group_samples = np.stack([sig[:local_len] for sig in channel_signals], axis=1)
			group_labels = np.full(local_len, label, dtype=np.uint8)

			grouped_samples[class_name].append(group_samples)
			grouped_labels[class_name].append(group_labels)

	max_groups = max(group_stats.values())
	class_order = list(class_label_map.keys())
	for group_idx in range(max_groups):
		for class_name in class_order:
			if group_idx < len(grouped_samples[class_name]):
				samples_list.append(grouped_samples[class_name][group_idx])
				labels_list.append(grouped_labels[class_name][group_idx])

	dataset_samples = np.concatenate(samples_list, axis=0)
	dataset_labels = np.concatenate(labels_list, axis=0)
	return dataset_samples, dataset_labels, group_stats


def normalize_to_int16(samples: np.ndarray) -> Tuple[np.ndarray, int, int]:
	min_val = int(samples.min())
	max_val = int(samples.max())

	if min_val == max_val:
		return np.zeros_like(samples, dtype=np.int16), min_val, max_val

	scaled = ((samples.astype(np.float64) - min_val) * 65535.0 / (max_val - min_val)) - 32768.0
	clipped = np.clip(np.round(scaled), -32768, 32767).astype(np.int16)
	return clipped, min_val, max_val


def _int16_to_hex(value: int) -> str:
	return f"{(int(value) & 0xFFFF):04X}"


def write_mem_files(samples_int16: np.ndarray, labels: np.ndarray, output_dir: Path) -> Tuple[Path, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	eeg_mem_path = output_dir / "eeg_dataset.mem"
	labels_mem_path = output_dir / "labels.mem"

	with eeg_mem_path.open("w", encoding="ascii") as eeg_file:
		for row in samples_int16:
			eeg_file.write(" ".join(_int16_to_hex(v) for v in row))
			eeg_file.write("\n")

	with labels_mem_path.open("w", encoding="ascii") as label_file:
		for label in labels:
			label_file.write(f"{int(label)}\n")

	return eeg_mem_path, labels_mem_path


def validate_mem_layout(eeg_mem_path: Path, labels_mem_path: Path, expected_samples: int) -> None:
	with eeg_mem_path.open("r", encoding="ascii") as eeg_file:
		eeg_lines = [line.strip() for line in eeg_file if line.strip()]

	with labels_mem_path.open("r", encoding="ascii") as labels_file:
		label_lines = [line.strip() for line in labels_file if line.strip()]

	if len(eeg_lines) != expected_samples:
		raise ValueError(f"eeg_dataset.mem line count mismatch: {len(eeg_lines)} vs {expected_samples}")

	if len(label_lines) != expected_samples:
		raise ValueError(f"labels.mem line count mismatch: {len(label_lines)} vs {expected_samples}")

	for idx, line in enumerate(eeg_lines[:32]):
		parts = line.split()
		if len(parts) != 4:
			raise ValueError(f"Invalid eeg_dataset.mem line {idx + 1}: expected 4 values, got {len(parts)}")
		for token in parts:
			int(token, 16)

	for idx, line in enumerate(label_lines[:64]):
		if line not in {"0", "1"}:
			raise ValueError(f"Invalid labels.mem line {idx + 1}: {line}")


def parse_args() -> argparse.Namespace:
	root = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(description="Generate eeg_dataset.mem and labels.mem from Bonn EEG dataset.")
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=root / "EEG-Dataset" / "Dataset",
		help="Path to Bonn dataset folder containing Z/O/N/S directories.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=root / "Epilepsy_Seizure_Detection_verilog",
		help="Directory where eeg_dataset.mem and labels.mem are written.",
	)
	parser.add_argument(
		"--classes",
		nargs="+",
		default=["Z", "O", "N", "S"],
		help="Classes to include. Use Z O N S for required pipeline.",
	)
	parser.add_argument(
		"--max-groups-per-class",
		type=int,
		default=None,
		help="Optional cap on number of 4-file groups per class.",
	)
	parser.add_argument(
		"--sample-length",
		type=int,
		default=None,
		help="Optional per-file sample limit before grouping.",
	)
	parser.add_argument(
		"--metadata-json",
		type=Path,
		default=None,
		help="Optional output path for dataset metadata JSON.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	class_map: Dict[str, int] = {}
	for cls in args.classes:
		key = cls.upper()
		if key not in CLASS_LABEL_MAP:
			raise ValueError(f"Unsupported class: {cls}. Supported classes: {sorted(CLASS_LABEL_MAP)}")
		class_map[key] = CLASS_LABEL_MAP[key]

	raw_samples, labels, group_stats = build_4channel_dataset(
		dataset_root=args.dataset_root,
		class_label_map=class_map,
		max_groups_per_class=args.max_groups_per_class,
		sample_length=args.sample_length,
	)

	normalized_samples, min_val, max_val = normalize_to_int16(raw_samples)
	eeg_mem_path, labels_mem_path = write_mem_files(normalized_samples, labels, args.output_dir)
	validate_mem_layout(eeg_mem_path, labels_mem_path, expected_samples=len(labels))

	seizure_count = int((labels == 1).sum())
	normal_count = int((labels == 0).sum())
	metadata = {
		"dataset_root": str(args.dataset_root),
		"output_dir": str(args.output_dir),
		"classes": class_map,
		"groups_used_per_class": group_stats,
		"samples_total": int(len(labels)),
		"normal_samples": normal_count,
		"seizure_samples": seizure_count,
		"raw_min": min_val,
		"raw_max": max_val,
	}

	if args.metadata_json is not None:
		args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
		with args.metadata_json.open("w", encoding="ascii") as metadata_file:
			json.dump(metadata, metadata_file, indent=2)

	print("EEG mem generation complete")
	print(f"  eeg mem: {eeg_mem_path}")
	print(f"  label mem: {labels_mem_path}")
	print(f"  total samples: {metadata['samples_total']}")
	print(f"  normal/seizure: {normal_count}/{seizure_count}")
	print(f"  raw min/max: {min_val}/{max_val}")


if __name__ == "__main__":
	main()
