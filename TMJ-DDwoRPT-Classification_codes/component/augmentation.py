import torchvision.transforms as transforms

VIT_MODELS = {"vit_b_16", "vit_b_32"}
CNN_MODELS = {"resnet18", "resnet50", "resnet101", "vgg11", "vgg16", "vgg19", "googlenet", "googlenet_clinical"}


def normalize_model_type(model_type: str) -> str:
    model_type = model_type.lower()
    if model_type.endswith("_sep"):
        model_type = model_type[:-4]
    return model_type


def get_image_size(scale=1, model_type="resnet101", input_type=None):
    base_model_type = normalize_model_type(model_type)

    if base_model_type in VIT_MODELS:
        return (224, 224)

    if base_model_type in CNN_MODELS:
        if input_type in ["combined", "combined_separated"]:
            return (int(1288 * scale), int(3232 * scale))
        return (int(1288 * scale), int(1616 * scale))

    raise ValueError(f"Unsupported model type: {model_type}")


def _build_rotation(level: str):
    if level == "off":
        return None
    if level == "base":
        return transforms.RandomRotation(15)
    if level == "x2":
        return transforms.RandomRotation(30)
    raise ValueError(f"Unsupported rotation_level: {level}")


def _build_color_jitter(level: str, base_model_type: str):
    if level == "off":
        return None

    if base_model_type in VIT_MODELS:
        if level == "base":
            return transforms.ColorJitter(
                brightness=(1.0, 1.2),
                contrast=1.2,
            )
        if level == "x2":
            return transforms.ColorJitter(
                brightness=(1.0, 1.4),
                contrast=1.4,
            )

    if base_model_type in CNN_MODELS:
        if level == "base":
            return transforms.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=1.2,
            )
        if level == "x2":
            return transforms.ColorJitter(
                brightness=(0.6, 1.4),
                contrast=1.4,
            )

    raise ValueError(f"Unsupported color_level: {level}")


def _build_horizontal_flip(flip_mode: str):
    if flip_mode == "off":
        return None
    if flip_mode == "on":
        return transforms.RandomHorizontalFlip(p=0.5)
    raise ValueError(f"Unsupported flip_mode: {flip_mode}")


def _build_sharpness(level: str, base_model_type: str):
    if level == "off":
        return None

    if base_model_type in VIT_MODELS:
        if level == "base":
            return transforms.RandomAdjustSharpness(1.2, p=0.5)
        if level == "x2":
            return transforms.RandomAdjustSharpness(2.4, p=0.5)

    if base_model_type in CNN_MODELS:
        if level == "base":
            return transforms.RandomAdjustSharpness(1.5, p=0.5)
        if level == "x2":
            return transforms.RandomAdjustSharpness(3.0, p=0.5)

    raise ValueError(f"Unsupported sharpness_level: {level}")


def _build_shear(level: str):
    if level == "off":
        return None
    if level == "base":
        return transforms.RandomAffine(
            degrees=0,
            shear=3,
        )
    if level == "x0.5":
        return transforms.RandomAffine(
            degrees=0,
            shear=0.5,
        )
    raise ValueError(f"Unsupported shear_level: {level}")



def get_augmentation(
    scale=1,
    model_type="resnet101",
    input_type=None,
    rotation_level="base",
    color_level="base",
    flip_mode="on",
    sharpness_level="base",
    shear_level="off",
):
    base_model_type = normalize_model_type(model_type)
    size = get_image_size(scale=scale, model_type=model_type, input_type=input_type)

    transform_list = [
        transforms.Resize(size),
    ]

    rotation_t = _build_rotation(rotation_level)
    if rotation_t is not None:
        transform_list.append(rotation_t)

    color_t = _build_color_jitter(color_level, base_model_type)
    if color_t is not None:
        transform_list.append(color_t)

    flip_t = _build_horizontal_flip(flip_mode)
    if flip_t is not None:
        transform_list.append(flip_t)

    sharpness_t = _build_sharpness(sharpness_level, base_model_type)
    if sharpness_t is not None:
        transform_list.append(sharpness_t)
    
    shear_t = _build_shear(shear_level)
    if shear_t is not None:
        transform_list.append(shear_t)

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)