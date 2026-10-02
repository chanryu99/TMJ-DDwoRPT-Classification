import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from component.load_data import (
    SingleSideDataset,
    FlippedLeftDataset,
    CombinedSideDataset,
    CombinedSideSeparatedDataset,
)
from component.model import get_model
from component.save_metric import MetricTracker, SeparateOutputMetricTracker
from component.augmentation import get_augmentation, get_image_size


def parse_args():
    parser = argparse.ArgumentParser(description="TMJ classification training")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument(
        "--input_type",
        type=str,
        required=True,
        choices=["single_left", "single_right", "flipped_left", "combined", "combined_separated"],
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=[
            "resnet18",
            "resnet50",
            "resnet101",
            "vgg11",
            "vgg16",
            "vgg19",
            "googlenet",
            "googlenet_clinical",
            "vit_b_16",
            "vit_b_32",
            "resnet18_sep",
            "resnet50_sep",
            "resnet101_sep",
            "vgg16_sep",
            "vgg19_sep",
            "googlenet_sep",
            "vit_b_16_sep",
            "vit_b_32_sep",
        ],
    )
    parser.add_argument("--image_dir", type=str, default="./preprocessed/cropped")
    parser.add_argument("--clinical_csv", type=str, default="./raw_data/clinical.csv")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--clinical_features",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--rotation_level",
        type=str,
        default="base",
        choices=["off", "base", "x2"],
    )
    parser.add_argument(
        "--color_level",
        type=str,
        default="base",
        choices=["off", "base", "x2"],
    )
    parser.add_argument(
        "--flip_mode",
        type=str,
        default="on",
        choices=["off", "on"],
    )
    parser.add_argument(
        "--sharpness_level",
        type=str,
        default="base",
        choices=["off", "base", "x2"],
    )
    parser.add_argument(
        "--shear_level",
        type=str,
        default="base",
        choices=["off", "base", "x0.5"],
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--imbalance_weight_loss", action="store_true")
    parser.add_argument("--minority_resample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./result",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_separate_mode(args) -> bool:
    return args.input_type == "combined_separated"


def uses_clinical_input(args) -> bool:
    return args.backbone == "googlenet_clinical"


def get_num_classes(task: int, input_type: str) -> int:
    if input_type in ["single_left", "single_right", "flipped_left", "combined_separated"]:
        if task in [1, 2, 3, 5, 6, 7]:
            return 2
        if task == 4:
            return 4
    if input_type == "combined":
        if task in [1, 2, 3, 5, 6, 7]:
            return 4
        if task == 4:
            return 16
    raise ValueError(
        f"Unsupported task/input_type combination: task={task}, input_type={input_type}"
    )


def build_dataset(args, transform):
    if args.input_type == "single_left":
        return SingleSideDataset(
            side="left",
            image_dir=args.image_dir,
            clinical_csv=args.clinical_csv,
            task=args.task,
            transform=transform,
        )
    if args.input_type == "single_right":
        return SingleSideDataset(
            side="right",
            image_dir=args.image_dir,
            clinical_csv=args.clinical_csv,
            task=args.task,
            transform=transform,
        )
    if args.input_type == "flipped_left":
        return FlippedLeftDataset(
            image_dir=args.image_dir,
            clinical_csv=args.clinical_csv,
            task=args.task,
            transform=transform,
            clinical_features=args.clinical_features,
        )
    if args.input_type == "combined":
        return CombinedSideDataset(
            image_dir=args.image_dir,
            clinical_csv=args.clinical_csv,
            task=args.task,
            transform=transform,
        )
    if args.input_type == "combined_separated":
        return CombinedSideSeparatedDataset(
            image_dir=args.image_dir,
            clinical_csv=args.clinical_csv,
            task=args.task,
            transform=transform,
        )
    raise ValueError(f"Unsupported input_type: {args.input_type}")


def save_config_txt(args):
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "config.txt")
    with open(save_path, "w", encoding="utf-8-sig") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def split_indices_by_patient_id(base_dataset, seed):
    indices = np.arange(len(base_dataset))
    groups = [sample["patient_id"] for sample in base_dataset.samples]
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
    train_idx, temp_idx = next(gss1.split(indices, groups=groups))
    temp_indices = indices[temp_idx]
    temp_groups = [groups[i] for i in temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_sub_idx, test_sub_idx = next(gss2.split(temp_indices, groups=temp_groups))
    val_idx = temp_indices[val_sub_idx]
    test_idx = temp_indices[test_sub_idx]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def compute_global_minmax_from_indices(base_dataset, indices):
    global_min = float("inf")
    global_max = float("-inf")
    for idx in indices:
        sample = base_dataset[idx]
        img = sample[0]
        img_min = img.min().item()
        img_max = img.max().item()
        if img_min < global_min:
            global_min = img_min
        if img_max > global_max:
            global_max = img_max
    if global_min == float("inf") or global_max == float("-inf"):
        raise ValueError("Failed to compute global min/max from training indices.")
    return global_min, global_max


class GlobalMinMaxNormalize:
    def __init__(self, global_min, global_max, eps=1e-8):
        self.global_min = float(global_min)
        self.global_max = float(global_max)
        self.eps = eps
    def __call__(self, tensor):
        denom = self.global_max - self.global_min
        if denom < self.eps:
            return tensor
        return (tensor - self.global_min) / (denom + self.eps)


def get_eval_transform(scale=1.0, model_type="resnet101", input_type=None, normalize=None):
    size = get_image_size(scale=scale, model_type=model_type, input_type=input_type)
    t = [
        transforms.Resize(size),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        t.append(normalize)
    return transforms.Compose(t)


def get_train_transform(
    scale=1.0,
    model_type="resnet101",
    input_type=None,
    normalize=None,
    rotation_level="base",
    color_level="base",
    flip_mode="on",
    sharpness_level="base",
    shear_level="off"
):
    aug = get_augmentation(
        scale=scale,
        model_type=model_type,
        input_type=input_type,
        rotation_level=rotation_level,
        color_level=color_level,
        flip_mode=flip_mode,
        sharpness_level=sharpness_level,
        shear_level=shear_level,
    )
    if isinstance(aug, transforms.Compose):
        aug_list = list(aug.transforms)
        if normalize is not None:
            aug_list.append(normalize)
        return transforms.Compose(aug_list)
    size = get_image_size(scale=scale, model_type=model_type, input_type=input_type)
    t = [
        transforms.Resize(size),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        t.append(normalize)
    return transforms.Compose(t)


def get_simple_class_weights(base_dataset, train_dataset, num_classes, device):
    labels = [base_dataset.samples[i]["label"] for i in train_dataset.indices]
    counts = [labels.count(c) for c in range(num_classes)]
    nonzero_counts = [c for c in counts if c > 0]
    if not nonzero_counts:
        raise ValueError("No training labels found while computing class weights.")
    min_count = min(nonzero_counts)
    weights = [min_count / c if c > 0 else 0.0 for c in counts]
    print(f"train class counts : {counts}")
    print(f"class weights      : {weights}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def get_simple_class_weights_separate(base_dataset, train_dataset, num_classes, device):
    right_labels = [base_dataset.samples[i]["right_label"] for i in train_dataset.indices]
    left_labels = [base_dataset.samples[i]["left_label"] for i in train_dataset.indices]
    all_labels = right_labels + left_labels
    counts = [all_labels.count(c) for c in range(num_classes)]
    nonzero_counts = [c for c in counts if c > 0]
    if not nonzero_counts:
        raise ValueError("No training labels found while computing separate class weights.")
    min_count = min(nonzero_counts)
    weights = [min_count / c if c > 0 else 0.0 for c in counts]
    print(f"train right class counts : {[right_labels.count(c) for c in range(num_classes)]}")
    print(f"train left class counts  : {[left_labels.count(c) for c in range(num_classes)]}")
    print(f"merged class counts      : {counts}")
    print(f"class weights            : {weights}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def resample_minority_to_majority(base_dataset, train_indices, num_classes, seed):
    rng = np.random.RandomState(seed)
    labels = [base_dataset.samples[i]["label"] for i in train_indices]
    class_to_indices = {c: [] for c in range(num_classes)}
    for idx, label in zip(train_indices, labels):
        class_to_indices[label].append(idx)
    counts = [len(class_to_indices[c]) for c in range(num_classes)]
    max_count = max(counts)
    resampled_indices = []
    for c in range(num_classes):
        cls_indices = class_to_indices[c]
        if len(cls_indices) == 0:
            continue
        if len(cls_indices) < max_count:
            extra_indices = rng.choice(
                cls_indices,
                size=max_count - len(cls_indices),
                replace=True,
            ).tolist()
            cls_indices = cls_indices + extra_indices
        resampled_indices.extend(cls_indices)
    rng.shuffle(resampled_indices)
    resampled_labels = [base_dataset.samples[i]["label"] for i in resampled_indices]
    resampled_counts = [resampled_labels.count(c) for c in range(num_classes)]
    print(f"original train class counts : {counts}")
    print(f"resampled class counts      : {resampled_counts}")
    return resampled_indices


def resample_minority_to_majority_separate(base_dataset, train_indices, num_classes, seed):
    rng = np.random.RandomState(seed)
    pair_ids = []
    for i in train_indices:
        r = base_dataset.samples[i]["right_label"]
        l = base_dataset.samples[i]["left_label"]
        pair_ids.append(r * num_classes + l)
    pair_class_to_indices = {}
    for idx, pair_id in zip(train_indices, pair_ids):
        if pair_id not in pair_class_to_indices:
            pair_class_to_indices[pair_id] = []
        pair_class_to_indices[pair_id].append(idx)
    counts = {k: len(v) for k, v in pair_class_to_indices.items()}
    max_count = max(counts.values())
    resampled_indices = []
    for _, cls_indices in pair_class_to_indices.items():
        if len(cls_indices) < max_count:
            extra_indices = rng.choice(
                cls_indices,
                size=max_count - len(cls_indices),
                replace=True,
            ).tolist()
            cls_indices = cls_indices + extra_indices
        resampled_indices.extend(cls_indices)
    rng.shuffle(resampled_indices)
    resampled_pair_ids = []
    for i in resampled_indices:
        r = base_dataset.samples[i]["right_label"]
        l = base_dataset.samples[i]["left_label"]
        resampled_pair_ids.append(r * num_classes + l)
    unique_pairs = sorted(set(resampled_pair_ids))
    resampled_counts = {k: resampled_pair_ids.count(k) for k in unique_pairs}
    print(f"original pair counts  : {counts}")
    print(f"resampled pair counts : {resampled_counts}")
    return resampled_indices


def build_splits_and_loaders(args):
    base_dataset = build_dataset(args, transform=None)
    train_indices, val_indices, test_indices = split_indices_by_patient_id(
        base_dataset=base_dataset,
        seed=args.seed,
    )
    global_min, global_max = compute_global_minmax_from_indices(
        base_dataset=base_dataset,
        indices=train_indices,
    )
    normalize = GlobalMinMaxNormalize(global_min=global_min, global_max=global_max)
    if args.minority_resample:
        if is_separate_mode(args):
            train_indices = resample_minority_to_majority_separate(
                base_dataset=base_dataset,
                train_indices=train_indices,
                num_classes=get_num_classes(args.task, args.input_type),
                seed=args.seed,
            )
        else:
            train_indices = resample_minority_to_majority(
                base_dataset=base_dataset,
                train_indices=train_indices,
                num_classes=get_num_classes(args.task, args.input_type),
                seed=args.seed,
            )
    train_transform = get_train_transform(
        scale=args.scale,
        model_type=args.backbone,
        input_type=args.input_type,
        normalize=normalize,
        rotation_level=args.rotation_level,
        color_level=args.color_level,
        flip_mode=args.flip_mode,
        sharpness_level=args.sharpness_level,
        shear_level=args.shear_level,
    )
    eval_transform = get_eval_transform(
        scale=args.scale,
        model_type=args.backbone,
        input_type=args.input_type,
        normalize=normalize,
    )
    train_dataset_full = build_dataset(args, transform=train_transform)
    val_dataset_full = build_dataset(args, transform=eval_transform)
    test_dataset_full = build_dataset(args, transform=eval_transform)
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    return base_dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, use_clinical=False):
    model.train()
    total_loss = 0.0
    all_true = []
    all_pred = []
    for batch in loader:
        if len(batch) == 3:
            imgs, clinicals, labels = batch
            imgs = imgs.to(device)
            clinicals = clinicals.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if use_clinical:
                outputs = model(imgs, clinicals)
            else:
                outputs = model(imgs)
        else:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_true, all_pred


def evaluate(model, loader, criterion, device, use_clinical=False):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, clinicals, labels = batch
                imgs = imgs.to(device)
                clinicals = clinicals.to(device)
                labels = labels.to(device)
                if use_clinical:
                    outputs = model(imgs, clinicals)
                else:
                    outputs = model(imgs)
            else:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_true, all_pred


def train_one_epoch_separate(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_true_right = []
    all_pred_right = []
    all_true_left = []
    all_pred_left = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        right_labels = labels[:, 0]
        left_labels = labels[:, 1]
        optimizer.zero_grad()
        right_outputs, left_outputs = model(imgs)
        loss_right = criterion(right_outputs, right_labels)
        loss_left = criterion(left_outputs, left_labels)
        loss = loss_right + loss_left
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        right_preds = right_outputs.argmax(dim=1)
        left_preds = left_outputs.argmax(dim=1)
        all_true_right.extend(right_labels.detach().cpu().tolist())
        all_pred_right.extend(right_preds.detach().cpu().tolist())
        all_true_left.extend(left_labels.detach().cpu().tolist())
        all_pred_left.extend(left_preds.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_true_right, all_pred_right, all_true_left, all_pred_left


def evaluate_separate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true_right = []
    all_pred_right = []
    all_true_left = []
    all_pred_left = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            right_labels = labels[:, 0]
            left_labels = labels[:, 1]
            right_outputs, left_outputs = model(imgs)
            loss_right = criterion(right_outputs, right_labels)
            loss_left = criterion(left_outputs, left_labels)
            loss = loss_right + loss_left
            total_loss += loss.item() * imgs.size(0)
            right_preds = right_outputs.argmax(dim=1)
            left_preds = left_outputs.argmax(dim=1)
            all_true_right.extend(right_labels.detach().cpu().tolist())
            all_pred_right.extend(right_preds.detach().cpu().tolist())
            all_true_left.extend(left_labels.detach().cpu().tolist())
            all_pred_left.extend(left_preds.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_true_right, all_pred_right, all_true_left, all_pred_left



def main():
    args = parse_args()
    set_seed(args.seed)
    if args.input_type == "combined_separated" and not args.backbone.endswith("_sep"):
        raise ValueError("combined_separated input_type requires a *_sep backbone.")
    if args.input_type != "combined_separated" and args.backbone.endswith("_sep"):
        raise ValueError("*_sep backbone can only be used with combined_separated input_type.")
    if args.backbone == "googlenet_clinical" and args.input_type != "flipped_left":
        raise ValueError("googlenet_clinical is intended to be used with flipped_left input_type.")
    if args.backbone == "googlenet_clinical" and len(args.clinical_features) == 0:
        raise ValueError("googlenet_clinical requires --clinical_features.")
    aug_str = (
        f"_rot-{args.rotation_level}"
        f"_color-{args.color_level}"
        f"_flip-{args.flip_mode}"
        f"_sharp-{args.sharpness_level}"
        f"_shear-{args.shear_level}"
    )
    pretrained_str = "pretrained" if args.pretrained else "scratch"
    imbalance_loss_str = "weightCE" if args.imbalance_weight_loss else "CE"
    resample_str = "_imbalance" if args.minority_resample else ""
    clinical_str = ""
    if len(args.clinical_features) > 0:
        clinical_str = "_clinical-" + "-".join(args.clinical_features)
    args.save_dir = os.path.join(
        args.save_dir,
        f"{args.task}_{args.input_type}_{args.seed}_{args.clinical_features}",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    save_config_txt(args)
    num_classes = get_num_classes(args.task, args.input_type)
    (
        base_dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_splits_and_loaders(args)

    use_clinical = uses_clinical_input(args)
    clinical_dim = None
    if use_clinical:
        if len(base_dataset.samples) == 0:
            raise ValueError("Dataset is empty, so clinical_dim cannot be inferred.")
        clinical_dim = len(base_dataset.samples[0]["clinical_feature"])
        print(f"clinical_dim      : {clinical_dim}")
    model = get_model(
        model_name=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        clinical_dim=clinical_dim,
    )
    model = model.to(device)
    if args.imbalance_weight_loss:
        if is_separate_mode(args):
            class_weights = get_simple_class_weights_separate(
                base_dataset=base_dataset,
                train_dataset=train_dataset,
                num_classes=num_classes,
                device=device,
            )
        else:
            class_weights = get_simple_class_weights(
                base_dataset=base_dataset,
                train_dataset=train_dataset,
                num_classes=num_classes,
                device=device,
            )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if is_separate_mode(args):
        tracker = SeparateOutputMetricTracker(
            save_dir=args.save_dir,
            num_classes=num_classes,
        )
    else:
        tracker = MetricTracker(
            save_dir=args.save_dir,
            num_classes=num_classes,
            decode_combined_pairs=(args.input_type == "combined"),
        )
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    print(f"device           : {device}")
    print(f"task             : {args.task}")
    print(f"input_type       : {args.input_type}")
    print(f"backbone         : {args.backbone}")
    print(f"pretrained       : {args.pretrained}")
    print(f"num_classes      : {num_classes}")
    print(f"save_dir         : {args.save_dir}")
    for epoch in range(1, args.epochs + 1):
        if is_separate_mode(args):
            train_loss, train_true_right, train_pred_right, train_true_left, train_pred_left = train_one_epoch_separate(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_true_right, val_pred_right, val_true_left, val_pred_left = evaluate_separate(
                model, val_loader, criterion, device
            )
            test_loss, test_true_right, test_pred_right, test_true_left, test_pred_left = evaluate_separate(
                model, test_loader, criterion, device
            )
            tracker.update(
                split="train",
                epoch=epoch,
                loss=train_loss,
                y_true_left=train_true_left,
                y_pred_left=train_pred_left,
                y_true_right=train_true_right,
                y_pred_right=train_pred_right,
            )
            tracker.update(
                split="val",
                epoch=epoch,
                loss=val_loss,
                y_true_left=val_true_left,
                y_pred_left=val_pred_left,
                y_true_right=val_true_right,
                y_pred_right=val_pred_right,
            )
            tracker.update(
                split="test",
                epoch=epoch,
                loss=test_loss,
                y_true_left=test_true_left,
                y_pred_left=test_pred_left,
                y_true_right=test_true_right,
                y_pred_right=test_pred_right,
            )
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} train_combined_f1={tracker.records[-3]['combined_f1_macro']:.4f} | "
                f"val_loss={val_loss:.4f} val_combined_f1={tracker.records[-2]['combined_f1_macro']:.4f} | "
                f"test_loss={test_loss:.4f} test_combined_f1={tracker.records[-1]['combined_f1_macro']:.4f}"
            )
        else:
            train_loss, train_true, train_pred = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_clinical=use_clinical,
            )
            val_loss, val_true, val_pred = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_clinical=use_clinical,
            )
            test_loss, test_true, test_pred = evaluate(
                model,
                test_loader,
                criterion,
                device,
                use_clinical=use_clinical,
            )
            tracker.update(
                split="train",
                epoch=epoch,
                loss=train_loss,
                y_true=train_true,
                y_pred=train_pred,
            )
            tracker.update(
                split="val",
                epoch=epoch,
                loss=val_loss,
                y_true=val_true,
                y_pred=val_pred,
            )
            tracker.update(
                split="test",
                epoch=epoch,
                loss=test_loss,
                y_true=test_true,
                y_pred=test_pred,
            )
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} train_f1={tracker.records[-3]['f1_macro']:.4f} | "
                f"val_loss={val_loss:.4f} val_f1={tracker.records[-2]['f1_macro']:.4f} | "
                f"test_loss={test_loss:.4f} test_f1={tracker.records[-1]['f1_macro']:.4f}"
            )
        if is_separate_mode(args):
            val_f1 = tracker.records[-2]["combined_f1_macro"]
        else:
            val_f1 = tracker.records[-2]["f1_macro"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> best model updated at epoch {epoch} (val_f1={val_f1:.4f})")
        if args.save_every_epoch:
            tracker.save_csv(filename="metrics.csv")
    tracker.save_csv(filename="metrics.csv")
    if is_separate_mode(args):
        tracker.save_best_csv(
            split="val",
            metric="combined_f1_macro",
            mode="max",
            filename="best_metric.csv",
        )
    else:
        tracker.save_best_csv(
            split="val",
            metric="f1_macro",
            mode="max",
            filename="best_metric.csv",
        )
    print("=" * 80)
    print("Training finished")
    print(f"best_epoch      : {best_epoch}")
    print(f"best_val_f1     : {best_val_f1:.4f}")
    print(f"best_model_path : {best_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
