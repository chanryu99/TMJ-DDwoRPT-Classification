import os
import glob
import csv
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


def _concat_horizontally(images: List[Image.Image]) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    if len(set(heights)) != 1:
        raise ValueError("All images must have the same height to concatenate.")
    total_width = sum(widths)
    height = heights[0]
    canvas = Image.new("L", (total_width, height))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def _parse_filename(path: str) -> Tuple[str, str, str, str]:
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {path}")
    patient_id = parts[0]
    timestamp = parts[1]
    side = parts[2]
    state = parts[3]
    if side not in ["left", "right"]:
        raise ValueError(f"Unexpected side in filename: {path}")
    if state not in ["close", "open"]:
        raise ValueError(f"Unexpected state in filename: {path}")
    return patient_id, timestamp, side, state


def _safe_strip(value):
    if value is None:
        return ""
    return str(value).strip()


def _load_clinical_rows(csv_path: str) -> Dict[str, Dict[str, str]]:
    clinical_dict = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = str(row["Patient ID"]).strip()
            clinical_dict[patient_id] = {k: _safe_strip(v) for k, v in row.items()}
    return clinical_dict


def _load_clinical_labels(csv_path: str) -> Dict[str, Dict[str, int]]:
    clinical_dict = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = str(row["Patient ID"]).strip()
            mri_rt = int(float(row["MRI_Rt"]))
            mri_lt = int(float(row["MRI_Lt"]))
            clinical_dict[patient_id] = {
                "MRI_Rt": mri_rt,
                "MRI_Lt": mri_lt,
            }
    return clinical_dict


def _single_task_label(raw_label: int, task: int) -> Optional[int]:

    if task == 1:
        return 0 if raw_label == 0 else 1
    elif task == 2:
        if raw_label == 1:
            return 0
        elif raw_label in [2, 3]:
            return 1
        return None
    elif task == 3:
        if raw_label == 0:
            return 0
        elif raw_label == 2:
            return 1
        return None
    elif task == 4:
        if raw_label in [0, 1, 2, 3]:
            return raw_label
        return None
    elif task == 5:
        if raw_label == 0:
            return 0
        elif raw_label == 1:
            return 1
        return None
    elif task == 6:
        # TaskA
        return 1 if raw_label == 2 else 0
    elif task == 7:
        # TaskB
        if raw_label in [0, 1]:
            return 0
        elif raw_label == 2:
            return 1
        return None
    else:
        raise ValueError(f"Unsupported task: {task}")


def _combined_task_label(right_label: int, left_label: int, task: int) -> Optional[int]:

    if task == 1:
        r = 0 if right_label == 0 else 1
        l = 0 if left_label == 0 else 1
        return r * 2 + l
    elif task == 2:
        if right_label == 0 or left_label == 0:
            return None
        r = 0 if right_label == 1 else 1
        l = 0 if left_label == 1 else 1
        return r * 2 + l
    elif task == 3:
        if right_label not in [0, 2] or left_label not in [0, 2]:
            return None
        r = 0 if right_label == 0 else 1
        l = 0 if left_label == 0 else 1
        return r * 2 + l
    elif task == 4:
        if right_label not in [0, 1, 2, 3] or left_label not in [0, 1, 2, 3]:
            return None
        return right_label * 4 + left_label
    elif task == 5:
        if right_label not in [0, 1] or left_label not in [0, 1]:
            return None
        return right_label * 2 + left_label
    elif task == 6:
        # TaskA
        r = 1 if right_label == 2 else 0
        l = 1 if left_label == 2 else 0
        return r * 2 + l
    elif task == 7:
        # TaskB
        if right_label == 3 or left_label == 3:
            return None
        r = 1 if right_label == 2 else 0
        l = 1 if left_label == 2 else 0
        return r * 2 + l
    else:
        raise ValueError(f"Unsupported task: {task}")


def _normalize_feature_name(feature_name: str) -> str:
    if feature_name.endswith("_Lt") or feature_name.endswith("_Rt"):
        return feature_name[:-3]
    return feature_name


def _resolve_clinical_column(row: Dict[str, str], feature_name: str, side: str) -> Tuple[str, str]:

    base = _normalize_feature_name(feature_name)
    lt_col = f"{base}_Lt"
    rt_col = f"{base}_Rt"
    has_lt = lt_col in row
    has_rt = rt_col in row
    has_common = base in row
    if has_lt and has_rt:
        chosen_col = lt_col if side == "left" else rt_col
        return base, chosen_col
    if has_common:
        return base, base
    raise ValueError(
        f"Clinical feature '{feature_name}' not found. "
        f"Expected either '{base}' or both '{lt_col}' and '{rt_col}' in clinical.csv."
    )


def _build_clinical_encoder(
    clinical_rows: Dict[str, Dict[str, str]],
    clinical_features: List[str],
) -> Dict[str, Dict]:

    if len(clinical_rows) == 0:
        return {}
    example_row = next(iter(clinical_rows.values()))
    encoder = {}
    for feature_name in clinical_features:
        base = _normalize_feature_name(feature_name)

        _, resolved_col_if_right = _resolve_clinical_column(example_row, feature_name, side="right")
        NUMERIC_FEATURES = {"AMO", "Age"}
        if base in NUMERIC_FEATURES:
            values = []

            if resolved_col_if_right == base:
                for row in clinical_rows.values():
                    v = _safe_strip(row.get(base, ""))
                    if v != "":
                        values.append(float(v))

            else:
                for row in clinical_rows.values():
                    for col in [f"{base}_Lt", f"{base}_Rt"]:
                        v = _safe_strip(row.get(col, ""))
                        if v != "":
                            values.append(float(v))
            if len(values) == 0:
                raise ValueError(
                    f"No valid numeric values found for {base}."
                )
            vmin = min(values)
            vmax = max(values)
            encoder[base] = {
                "type": "numeric",
                "min": vmin,
                "max": vmax,
            }
        else:
            categories = set()
            if resolved_col_if_right == base:
                for row in clinical_rows.values():
                    v = _safe_strip(row.get(base, ""))
                    if v == "":
                        v = "__MISSING__"
                    categories.add(v)
            else:
                for row in clinical_rows.values():
                    for col in [f"{base}_Lt", f"{base}_Rt"]:
                        v = _safe_strip(row.get(col, ""))
                        if v == "":
                            v = "__MISSING__"
                        categories.add(v)
            categories = sorted(categories)
            encoder[base] = {
                "type": "categorical",
                "categories": categories,
                "cat2idx": {cat: i for i, cat in enumerate(categories)},
            }
    return encoder


def _encode_clinical_features(
    row: Dict[str, str],
    side: str,
    clinical_features: List[str],
    encoder: Dict[str, Dict],
) -> np.ndarray:
    encoded = []
    for feature_name in clinical_features:
        base, col = _resolve_clinical_column(row, feature_name, side)
        value = _safe_strip(row.get(col, ""))
        spec = encoder[base]
        if spec["type"] == "numeric":
            if value == "":
                scaled = 0.0
            else:
                x = float(value)
                vmin = spec["min"]
                vmax = spec["max"]
                if vmax > vmin:
                    scaled = (x - vmin) / (vmax - vmin)
                else:
                    scaled = 0.0
            encoded.append(float(scaled))
        else:
            if value == "":
                value = "__MISSING__"
            one_hot = np.zeros(len(spec["categories"]), dtype=np.float32)
            idx = spec["cat2idx"][value]
            one_hot[idx] = 1.0
            encoded.extend(one_hot.tolist())
    return np.asarray(encoded, dtype=np.float32)


class DualimagesinglesideDataset(Dataset):


    def __init__(
        self,
        side: str,
        image_dir: str = "./preprocessed/cropped",
        clinical_csv: str = "./raw_data/clinical.csv",
        task: int = 1,
        transform=None,
        clinical_features: Optional[List[str]] = None,
    ):
        if side not in ["left", "right"]:
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.image_dir = image_dir
        self.clinical_csv = clinical_csv
        self.task = task
        self.transform = transform
        self.clinical_features = clinical_features or []
        self.clinical_dict = _load_clinical_labels(self.clinical_csv)
        self.clinical_rows = _load_clinical_rows(self.clinical_csv)
        self.clinical_encoder = _build_clinical_encoder(
            self.clinical_rows,
            self.clinical_features,
        )
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        grouped = {}
        for path in image_paths:
            patient_id, timestamp, side, state = _parse_filename(path)
            key = f"{patient_id}_{timestamp}"
            if key not in grouped:
                grouped[key] = {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "left_close": None,
                    "left_open": None,
                    "right_close": None,
                    "right_open": None,
                }
            grouped[key][f"{side}_{state}"] = path
        samples = []
        for key, item in grouped.items():
            if (
                item["left_close"] is None
                or item["left_open"] is None
                or item["right_close"] is None
                or item["right_open"] is None
            ):
                continue
            patient_id = item["patient_id"]
            if patient_id not in self.clinical_dict:
                continue
            if patient_id not in self.clinical_rows:
                continue
            if self.side == "left":
                raw_label = self.clinical_dict[patient_id]["MRI_Lt"]
            else:
                raw_label = self.clinical_dict[patient_id]["MRI_Rt"]
            label = _single_task_label(raw_label, self.task)
            if label is None:
                continue
            clinical_feature = _encode_clinical_features(
                row=self.clinical_rows[patient_id],
                side=self.side,
                clinical_features=self.clinical_features,
                encoder=self.clinical_encoder,
            )
            samples.append(
                {
                    "sample_id": key,
                    "patient_id": patient_id,
                    "side": self.side,
                    "right_close_path": item["right_close"],
                    "right_open_path": item["right_open"],
                    "left_open_path": item["left_open"],
                    "left_close_path": item["left_close"],
                    "raw_label": raw_label,
                    "label": label,
                    "clinical_feature": clinical_feature,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        right_close = Image.open(sample["right_close_path"]).convert("L")
        right_open = Image.open(sample["right_open_path"]).convert("L")
        left_open = Image.open(sample["left_open_path"]).convert("L")
        left_close = Image.open(sample["left_close_path"]).convert("L")
        concat_img = _concat_horizontally(
            [right_close, right_open, left_open, left_close]
        )
        if self.transform is not None:
            img = self.transform(concat_img)
        else:
            img = _pil_to_tensor(concat_img)
        clinical = torch.tensor(sample["clinical_feature"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, clinical, label


class SingleSideDataset(Dataset):

    def __init__(
        self,
        side: str,
        image_dir: str = "./preprocessed/cropped",
        clinical_csv: str = "./raw_data/clinical.csv",
        task: int = 1,
        transform=None,
        clinical_features: Optional[List[str]] = None,
    ):
        if side not in ["left", "right"]:
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.image_dir = image_dir
        self.clinical_csv = clinical_csv
        self.task = task
        self.transform = transform
        self.clinical_features = clinical_features or []
        self.clinical_dict = _load_clinical_labels(self.clinical_csv)
        self.clinical_rows = _load_clinical_rows(self.clinical_csv)
        self.clinical_encoder = _build_clinical_encoder(
            self.clinical_rows,
            self.clinical_features,
        )
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        grouped = {}
        for path in image_paths:
            patient_id, timestamp, side, state = _parse_filename(path)
            if side != self.side:
                continue
            key = f"{patient_id}_{timestamp}_{side}"
            if key not in grouped:
                grouped[key] = {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "side": side,
                    "close": None,
                    "open": None,
                }
            grouped[key][state] = path
        samples = []
        for key, item in grouped.items():
            if item["close"] is None or item["open"] is None:
                continue
            patient_id = item["patient_id"]
            if patient_id not in self.clinical_dict:
                continue
            if patient_id not in self.clinical_rows:
                continue
            if self.side == "left":
                raw_label = self.clinical_dict[patient_id]["MRI_Lt"]
            else:
                raw_label = self.clinical_dict[patient_id]["MRI_Rt"]
            label = _single_task_label(raw_label, self.task)
            if label is None:
                continue
            clinical_feature = _encode_clinical_features(
                row=self.clinical_rows[patient_id],
                side=self.side,
                clinical_features=self.clinical_features,
                encoder=self.clinical_encoder,
            )
            samples.append(
                {
                    "sample_id": key,
                    "patient_id": patient_id,
                    "side": self.side,
                    "close_path": item["close"],
                    "open_path": item["open"],
                    "raw_label": raw_label,
                    "label": label,
                    "clinical_feature": clinical_feature,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        close_img = Image.open(sample["close_path"]).convert("L")
        open_img = Image.open(sample["open_path"]).convert("L")
        concat_img = _concat_horizontally([close_img, open_img])
        if self.transform is not None:
            img = self.transform(concat_img)
        else:
            img = _pil_to_tensor(concat_img)
        clinical = torch.tensor(sample["clinical_feature"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, clinical, label


class FlippedLeftDataset(Dataset):

    def __init__(
        self,
        image_dir: str = "./preprocessed/cropped",
        clinical_csv: str = "./raw_data/clinical.csv",
        task: int = 1,
        transform=None,
        clinical_features: Optional[List[str]] = None,
    ):
        self.image_dir = image_dir
        self.clinical_csv = clinical_csv
        self.task = task
        self.transform = transform
        self.clinical_features = clinical_features or []

        self.clinical_dict = _load_clinical_labels(self.clinical_csv)

        self.clinical_rows = _load_clinical_rows(self.clinical_csv)

        self.clinical_encoder = _build_clinical_encoder(
            self.clinical_rows,
            self.clinical_features,
        )
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        grouped = {}
        for path in image_paths:
            patient_id, timestamp, side, state = _parse_filename(path)
            key = f"{patient_id}_{timestamp}_{side}"
            if key not in grouped:
                grouped[key] = {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "side": side,
                    "close": None,
                    "open": None,
                }
            grouped[key][state] = path
        samples = []
        for key, item in grouped.items():
            if item["close"] is None or item["open"] is None:
                continue
            patient_id = item["patient_id"]
            side = item["side"]
            if patient_id not in self.clinical_dict:
                continue
            if patient_id not in self.clinical_rows:
                continue
            raw_label = (
                self.clinical_dict[patient_id]["MRI_Lt"]
                if side == "left"
                else self.clinical_dict[patient_id]["MRI_Rt"]
            )
            label = _single_task_label(raw_label, self.task)
            if label is None:
                continue
            clinical_feature = _encode_clinical_features(
                row=self.clinical_rows[patient_id],
                side=side,
                clinical_features=self.clinical_features,
                encoder=self.clinical_encoder,
            )
            samples.append(
                {
                    "sample_id": key,
                    "patient_id": patient_id,
                    "side": side,
                    "close_path": item["close"],
                    "open_path": item["open"],
                    "raw_label": raw_label,
                    "label": label,
                    "clinical_feature": clinical_feature,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        close_img = Image.open(sample["close_path"]).convert("L")
        open_img = Image.open(sample["open_path"]).convert("L")
        if sample["side"] == "left":
            close_img = close_img.transpose(Image.FLIP_LEFT_RIGHT)
            open_img = open_img.transpose(Image.FLIP_LEFT_RIGHT)
        concat_img = _concat_horizontally([close_img, open_img])
        if self.transform is not None:
            img = self.transform(concat_img)
        else:
            img = _pil_to_tensor(concat_img)
        clinical = torch.tensor(sample["clinical_feature"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, clinical, label


class CombinedSideDataset(Dataset):

    def __init__(
        self,
        image_dir: str = "./preprocessed/cropped",
        clinical_csv: str = "./raw_data/clinical.csv",
        task: int = 1,
        transform=None,
    ):
        self.image_dir = image_dir
        self.clinical_csv = clinical_csv
        self.task = task
        self.transform = transform
        self.clinical_dict = _load_clinical_labels(self.clinical_csv)
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        grouped = {}
        for path in image_paths:
            patient_id, timestamp, side, state = _parse_filename(path)
            key = f"{patient_id}_{timestamp}"
            if key not in grouped:
                grouped[key] = {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "left_close": None,
                    "left_open": None,
                    "right_close": None,
                    "right_open": None,
                }
            grouped[key][f"{side}_{state}"] = path
        samples = []
        for key, item in grouped.items():
            if (
                item["left_close"] is None
                or item["left_open"] is None
                or item["right_close"] is None
                or item["right_open"] is None
            ):
                continue
            patient_id = item["patient_id"]
            if patient_id not in self.clinical_dict:
                continue
            right_label = self.clinical_dict[patient_id]["MRI_Rt"]
            left_label = self.clinical_dict[patient_id]["MRI_Lt"]
            label = _combined_task_label(right_label, left_label, self.task)
            if label is None:
                continue
            samples.append(
                {
                    "sample_id": key,
                    "patient_id": patient_id,
                    "right_close_path": item["right_close"],
                    "right_open_path": item["right_open"],
                    "left_open_path": item["left_open"],
                    "left_close_path": item["left_close"],
                    "right_raw_label": right_label,
                    "left_raw_label": left_label,
                    "label": label,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        right_close = Image.open(sample["right_close_path"]).convert("L")
        right_open = Image.open(sample["right_open_path"]).convert("L")
        left_open = Image.open(sample["left_open_path"]).convert("L")
        left_close = Image.open(sample["left_close_path"]).convert("L")
        concat_img = _concat_horizontally(
            [right_close, right_open, left_open, left_close]
        )
        if self.transform is not None:
            img = self.transform(concat_img)
        else:
            img = _pil_to_tensor(concat_img)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, label


class CombinedSideSeparatedDataset(Dataset):

    def __init__(
        self,
        image_dir: str = "./preprocessed/cropped",
        clinical_csv: str = "./raw_data/clinical.csv",
        task: int = 4,
        transform=None,
    ):
        self.image_dir = image_dir
        self.clinical_csv = clinical_csv
        self.task = task
        self.transform = transform
        self.clinical_dict = _load_clinical_labels(self.clinical_csv)
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        grouped = {}
        for path in image_paths:
            patient_id, timestamp, side, state = _parse_filename(path)
            key = f"{patient_id}_{timestamp}"
            if key not in grouped:
                grouped[key] = {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "left_close": None,
                    "left_open": None,
                    "right_close": None,
                    "right_open": None,
                }
            grouped[key][f"{side}_{state}"] = path
        samples = []
        for key, item in grouped.items():
            if (
                item["left_close"] is None
                or item["left_open"] is None
                or item["right_close"] is None
                or item["right_open"] is None
            ):
                continue
            patient_id = item["patient_id"]
            if patient_id not in self.clinical_dict:
                continue
            raw_right_label = self.clinical_dict[patient_id]["MRI_Rt"]
            raw_left_label = self.clinical_dict[patient_id]["MRI_Lt"]
            right_label = _single_task_label(raw_right_label, self.task)
            left_label = _single_task_label(raw_left_label, self.task)
            if right_label is None or left_label is None:
                continue
            samples.append(
                {
                    "sample_id": key,
                    "patient_id": patient_id,
                    "right_close_path": item["right_close"],
                    "right_open_path": item["right_open"],
                    "left_open_path": item["left_open"],
                    "left_close_path": item["left_close"],
                    "right_raw_label": raw_right_label,
                    "left_raw_label": raw_left_label,
                    "right_label": right_label,
                    "left_label": left_label,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        right_close = Image.open(sample["right_close_path"]).convert("L")
        right_open = Image.open(sample["right_open_path"]).convert("L")
        left_open = Image.open(sample["left_open_path"]).convert("L")
        left_close = Image.open(sample["left_close_path"]).convert("L")
        concat_img = _concat_horizontally(
            [right_close, right_open, left_open, left_close]
        )
        if self.transform is not None:
            img = self.transform(concat_img)
        else:
            img = _pil_to_tensor(concat_img)
        label = torch.tensor(
            [sample["right_label"], sample["left_label"]],
            dtype=torch.long,
        )
        return img, label
