"""PyTorch data loading for the UCI HAR inertial-signal windows."""
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


SIGNALS = (
	"total_acc_x",
	"total_acc_y",
	"total_acc_z",
	"body_acc_x",
	"body_acc_y",
	"body_acc_z",
	"body_gyro_x",
	"body_gyro_y",
	"body_gyro_z",
)

#each row is a window containing 128 readings
#each window has 50% overlap with the previous

class HARInertialDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
	"""UCI HAR windows with shape ``(128, 9)`` and zero-based labels."""

	def __init__(self, data_root: str | Path, split: str, *, augment: bool = False, noise_std: float = 0.02) -> None:
		if split not in {"train", "test"}:
			raise ValueError("split must be 'train' or 'test'")

		self.augment = augment
		self.noise_std = noise_std

		split_dir = Path(data_root) / split
		signal_dir = split_dir / "Inertial Signals"
		signal_arrays = []
		for signal in SIGNALS:
			path = signal_dir / f"{signal}_{split}.txt"
			#(N, 128)
			signal_array = np.loadtxt(path, dtype=np.float32)
			if signal_array.ndim == 1:
				#(1, 128)
				signal_array = signal_array.reshape(1, -1)
			signal_arrays.append(signal_array)

		sample_count = signal_arrays[0].shape[0]
		if any(array.shape != (sample_count, 128) for array in signal_arrays):
			raise ValueError("Every inertial signal file must have shape (N, 128)")

		# Stack nine channels = (N, 128, 9)
		self.windows = torch.from_numpy(np.stack(signal_arrays, axis=-1))
		# Convert labels from 1-6 to zero based 0-5 (N, )
		self.labels = torch.from_numpy(
			np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
		)
		# Subject identifier for each window, used to create validation splits: (N,).
		self.subjects = torch.from_numpy(
			np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)
		)

		if not len(self.windows) == len(self.labels) == len(self.subjects):
			raise ValueError("Signal, label, and subject counts do not match")

	def __len__(self) -> int:
		return self.windows.shape[0]

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		window = self.windows[index]
		if self.augment:
			#fresh noise per fetch, so the same window looks slightly different each epoch
			window = window + torch.randn_like(window) * self.noise_std
		return window, self.labels[index]

	def indices_for_subjects(self, subjects: Sequence[int]) -> list[int]:
		subject_set = {int(subject) for subject in subjects}
		return [
			index
			for index, subject in enumerate(self.subjects.tolist())
			if subject in subject_set
		]


def create_dataloaders(
	data_root: str | Path = "UCI HAR Dataset",
	*,
	batch_size: int = 64,
	validation_subjects: Sequence[int] = (3, 5, 7),
	num_workers: int = 0,
	add_noise: bool=False
) -> tuple[DataLoader, DataLoader, DataLoader]:
	"""Create train, validation, and official test loaders.

	Validation samples are selected by subject from the official training
	split. The test loader always uses the official test split.
	"""
	if batch_size < 1:
		raise ValueError("batch_size must be a positive integer")

	#separate instances so noise (training-only) can never leak into the validation split
	train_dataset = HARInertialDataset(data_root, "train", augment=add_noise)
	val_dataset = HARInertialDataset(data_root, "train", augment=False)
	test_dataset = HARInertialDataset(data_root, "test")

	validation_indices = val_dataset.indices_for_subjects(validation_subjects)
	validation_index_set = set(validation_indices)
	train_indices = [
		index for index in range(len(train_dataset)) if index not in validation_index_set
	]
	if not validation_indices or not train_indices:
		raise ValueError("validation_subjects must leave both train and validation data")

	return (
		DataLoader(
			Subset(train_dataset, train_indices),
			batch_size=batch_size,
			shuffle=True,
			num_workers=num_workers,
		),
		DataLoader(
			Subset(val_dataset, validation_indices),
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
		),
		DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
		),
	)
