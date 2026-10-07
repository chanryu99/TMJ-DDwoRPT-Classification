import os
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def _compute_classification_metrics(y_true, y_pred, num_classes: int) -> Dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0,
    )

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    result = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }

    for class_idx in range(num_classes):
        result[f"precision_class_{class_idx}"] = float(precision_per_class[class_idx])
        result[f"recall_class_{class_idx}"] = float(recall_per_class[class_idx])
        result[f"f1_class_{class_idx}"] = float(f1_per_class[class_idx])
        result[f"support_class_{class_idx}"] = int(support_per_class[class_idx])

    for i in range(num_classes):
        for j in range(num_classes):
            result[f"cm_{i}_{j}"] = int(cm[i, j])

    return result


class MetricTracker:

    def __init__(self, save_dir: str, num_classes: int, decode_combined_pairs: bool = False):
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.decode_combined_pairs = decode_combined_pairs
        os.makedirs(self.save_dir, exist_ok=True)

        self.records: List[Dict] = []

        if self.decode_combined_pairs:
            self.pair_base_classes = self._infer_pair_base_classes(num_classes)
            self.num_pair_classes = num_classes
        else:
            self.pair_base_classes = None
            self.num_pair_classes = None

    @staticmethod
    def _infer_pair_base_classes(num_classes: int) -> int:

        base = int(round(np.sqrt(num_classes)))
        if base * base != num_classes:
            raise ValueError(
                f"decode_combined_pairs=True requires square num_classes, got {num_classes}"
            )
        return base

    def _decode_pair_labels(self, y) -> Tuple[np.ndarray, np.ndarray]:

        y = np.array(y, dtype=int)
        base = self.pair_base_classes

        right = y // base
        left = y % base

        return right, left

    @staticmethod
    def _append_metric_fieldnames(fieldnames: List[str], prefix: str, num_classes: int):
        fieldnames.extend([
            f"{prefix}accuracy",
            f"{prefix}precision_macro",
            f"{prefix}recall_macro",
            f"{prefix}f1_macro",
        ])

        for class_idx in range(num_classes):
            fieldnames.extend([
                f"{prefix}precision_class_{class_idx}",
                f"{prefix}recall_class_{class_idx}",
                f"{prefix}f1_class_{class_idx}",
                f"{prefix}support_class_{class_idx}",
            ])

        for i in range(num_classes):
            for j in range(num_classes):
                fieldnames.append(f"{prefix}cm_{i}_{j}")

    def update(
        self,
        split: str,
        epoch: int,
        loss: float,
        y_true,
        y_pred,
    ):

        metric_result = _compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=self.num_classes,
        )

        record = {
            "epoch": int(epoch),
            "split": split,
            "loss": float(loss),
            **metric_result,
        }

        if self.decode_combined_pairs:
            y_true_right, y_true_left = self._decode_pair_labels(y_true)
            y_pred_right, y_pred_left = self._decode_pair_labels(y_pred)

            # right metric
            right_metrics = _compute_classification_metrics(
                y_true=y_true_right,
                y_pred=y_pred_right,
                num_classes=self.pair_base_classes,
            )

            # left metric
            left_metrics = _compute_classification_metrics(
                y_true=y_true_left,
                y_pred=y_pred_left,
                num_classes=self.pair_base_classes,
            )

            # combined metric
            y_true_combined = np.concatenate([y_true_left, y_true_right], axis=0)
            y_pred_combined = np.concatenate([y_pred_left, y_pred_right], axis=0)

            combined_metrics = _compute_classification_metrics(
                y_true=y_true_combined,
                y_pred=y_pred_combined,
                num_classes=self.pair_base_classes,
            )

            # both metric
            both_metrics = metric_result.copy()


            both_correct_accuracy = float(
                np.mean((y_true_left == y_pred_left) & (y_true_right == y_pred_right))
            )

            record["both_correct_accuracy"] = both_correct_accuracy

            for k, v in left_metrics.items():
                record[f"left_{k}"] = v

            for k, v in right_metrics.items():
                record[f"right_{k}"] = v

            for k, v in combined_metrics.items():
                record[f"combined_{k}"] = v

            for k, v in both_metrics.items():
                record[f"both_{k}"] = v

        self.records.append(record)

    def _get_fieldnames(self) -> List[str]:
        fieldnames = [
            "epoch",
            "split",
            "loss",
        ]


        self._append_metric_fieldnames(fieldnames, prefix="", num_classes=self.num_classes)

        if self.decode_combined_pairs:
            fieldnames.append("both_correct_accuracy")

            # both (pair-level)
            self._append_metric_fieldnames(
                fieldnames,
                prefix="both_",
                num_classes=self.num_classes,
            )

            # left
            self._append_metric_fieldnames(
                fieldnames,
                prefix="left_",
                num_classes=self.pair_base_classes,
            )

            # right
            self._append_metric_fieldnames(
                fieldnames,
                prefix="right_",
                num_classes=self.pair_base_classes,
            )

            # combined
            self._append_metric_fieldnames(
                fieldnames,
                prefix="combined_",
                num_classes=self.pair_base_classes,
            )

        return fieldnames

    def save_csv(self, filename: str = "metrics.csv"):
        if len(self.records) == 0:
            return

        save_path = os.path.join(self.save_dir, filename)
        fieldnames = self._get_fieldnames()

        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def get_best_epoch(self, split: str = "val", metric: str = "f1_macro", mode: str = "max") -> Optional[Dict]:
        target_records = [r for r in self.records if r["split"] == split]
        if len(target_records) == 0:
            return None

        if metric not in target_records[0]:
            raise KeyError(f"metric '{metric}' not found in records.")

        if mode == "max":
            best_record = max(target_records, key=lambda x: x[metric])
        elif mode == "min":
            best_record = min(target_records, key=lambda x: x[metric])
        else:
            raise ValueError("mode must be 'max' or 'min'")

        return best_record

    def save_best_csv(
        self,
        split: str = "val",
        metric: str = "f1_macro",
        mode: str = "max",
        filename: str = "best_metric.csv",
    ):
        best_record = self.get_best_epoch(split=split, metric=metric, mode=mode)
        if best_record is None:
            return

        best_epoch = best_record["epoch"]

        rows_to_save = []
        for target_split in ["val", "test"]:
            record = next(
                (r for r in self.records if r["epoch"] == best_epoch and r["split"] == target_split),
                None
            )
            if record is not None:
                rows_to_save.append(record)

        if len(rows_to_save) == 0:
            return

        save_path = os.path.join(self.save_dir, filename)
        fieldnames = self._get_fieldnames()

        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_save)


class SeparateOutputMetricTracker:

    def __init__(self, save_dir: str, num_classes: int):
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.num_pair_classes = num_classes * num_classes
        os.makedirs(self.save_dir, exist_ok=True)

        self.records: List[Dict] = []

    @staticmethod
    def _append_metric_fieldnames(fieldnames: List[str], prefix: str, num_classes: int):
        fieldnames.extend([
            f"{prefix}accuracy",
            f"{prefix}precision_macro",
            f"{prefix}recall_macro",
            f"{prefix}f1_macro",
        ])

        for class_idx in range(num_classes):
            fieldnames.extend([
                f"{prefix}precision_class_{class_idx}",
                f"{prefix}recall_class_{class_idx}",
                f"{prefix}f1_class_{class_idx}",
                f"{prefix}support_class_{class_idx}",
            ])

        for i in range(num_classes):
            for j in range(num_classes):
                fieldnames.append(f"{prefix}cm_{i}_{j}")

    def update(
        self,
        split: str,
        epoch: int,
        loss: float,
        y_true_left,
        y_pred_left,
        y_true_right,
        y_pred_right,
    ):
        y_true_left = np.array(y_true_left)
        y_pred_left = np.array(y_pred_left)
        y_true_right = np.array(y_true_right)
        y_pred_right = np.array(y_pred_right)

        if not (
            len(y_true_left) == len(y_pred_left) == len(y_true_right) == len(y_pred_right)
        ):
            raise ValueError("Left/right y_true and y_pred lengths must all match.")

        left_metrics = _compute_classification_metrics(
            y_true=y_true_left,
            y_pred=y_pred_left,
            num_classes=self.num_classes,
        )

        right_metrics = _compute_classification_metrics(
            y_true=y_true_right,
            y_pred=y_pred_right,
            num_classes=self.num_classes,
        )

        y_true_combined = np.concatenate([y_true_left, y_true_right], axis=0)
        y_pred_combined = np.concatenate([y_pred_left, y_pred_right], axis=0)

        combined_metrics = _compute_classification_metrics(
            y_true=y_true_combined,
            y_pred=y_pred_combined,
            num_classes=self.num_classes,
        )

        y_true_pair = y_true_left * self.num_classes + y_true_right
        y_pred_pair = y_pred_left * self.num_classes + y_pred_right

        pair_metrics = _compute_classification_metrics(
            y_true=y_true_pair,
            y_pred=y_pred_pair,
            num_classes=self.num_pair_classes,
        )

        both_correct_accuracy = float(
            np.mean((y_true_left == y_pred_left) & (y_true_right == y_pred_right))
        )

        record = {
            "epoch": int(epoch),
            "split": split,
            "loss": float(loss),
            "both_correct_accuracy": both_correct_accuracy,
        }

        for k, v in left_metrics.items():
            record[f"left_{k}"] = v

        for k, v in right_metrics.items():
            record[f"right_{k}"] = v

        for k, v in combined_metrics.items():
            record[f"combined_{k}"] = v

        for k, v in pair_metrics.items():
            record[f"both_{k}"] = v

        self.records.append(record)

    def _get_fieldnames(self) -> List[str]:
        fieldnames = [
            "epoch",
            "split",
            "loss",
            "both_correct_accuracy",
        ]

        # both
        self._append_metric_fieldnames(
            fieldnames,
            prefix="both_",
            num_classes=self.num_pair_classes,
        )

        # left
        self._append_metric_fieldnames(
            fieldnames,
            prefix="left_",
            num_classes=self.num_classes,
        )

        # right
        self._append_metric_fieldnames(
            fieldnames,
            prefix="right_",
            num_classes=self.num_classes,
        )

        # combined
        self._append_metric_fieldnames(
            fieldnames,
            prefix="combined_",
            num_classes=self.num_classes,
        )

        return fieldnames

    def save_csv(self, filename: str = "metrics.csv"):
        if len(self.records) == 0:
            return

        save_path = os.path.join(self.save_dir, filename)
        fieldnames = self._get_fieldnames()

        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def get_best_epoch(
        self,
        split: str = "val",
        metric: str = "combined_f1_macro",
        mode: str = "max"
    ) -> Optional[Dict]:
        target_records = [r for r in self.records if r["split"] == split]
        if len(target_records) == 0:
            return None

        if metric not in target_records[0]:
            raise KeyError(f"metric '{metric}' not found in records.")

        if mode == "max":
            best_record = max(target_records, key=lambda x: x[metric])
        elif mode == "min":
            best_record = min(target_records, key=lambda x: x[metric])
        else:
            raise ValueError("mode must be 'max' or 'min'")

        return best_record

    def save_best_csv(
        self,
        split: str = "val",
        metric: str = "combined_f1_macro",
        mode: str = "max",
        filename: str = "best_metric.csv",
    ):
        best_record = self.get_best_epoch(split=split, metric=metric, mode=mode)
        if best_record is None:
            return

        best_epoch = best_record["epoch"]

        rows_to_save = []
        for target_split in ["val", "test"]:
            record = next(
                (r for r in self.records if r["epoch"] == best_epoch and r["split"] == target_split),
                None
            )
            if record is not None:
                rows_to_save.append(record)

        if len(rows_to_save) == 0:
            return

        save_path = os.path.join(self.save_dir, filename)
        fieldnames = self._get_fieldnames()

        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_save)