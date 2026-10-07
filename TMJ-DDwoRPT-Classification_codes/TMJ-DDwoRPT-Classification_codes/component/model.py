import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet101(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        self.model = models.resnet101(weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG11(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.VGG11_Weights.DEFAULT if pretrained else None
        self.model = models.vgg11(weights=weights)
        self.model.features[0] = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1
        )
        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.model = models.vgg16(weights=weights)
        self.model.features[0] = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1
        )
        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


class VGG19(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        self.model = models.vgg19(weights=weights)
        self.model.features[0] = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1
        )
        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


class GoogleNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
        if pretrained:
            self.model = models.googlenet(
                weights=weights,
                aux_logits=True,
                transform_input=False,
                init_weights=False
            )
            self.model.aux_logits = False
            self.model.aux1 = None
            self.model.aux2 = None
        else:
            self.model = models.googlenet(
                weights=None,
                aux_logits=False,
                transform_input=False,
                init_weights=False
            )
        self.model.conv1.conv = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ViTB16(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_16(weights=weights)
        self.model.conv_proj = nn.Conv2d(
            1,
            self.model.conv_proj.out_channels,
            kernel_size=self.model.conv_proj.kernel_size,
            stride=self.model.conv_proj.stride,
            padding=self.model.conv_proj.padding,
            bias=(self.model.conv_proj.bias is not None),
        )
        self.model.heads.head = nn.Linear(
            self.model.heads.head.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


class ViTB32(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_32(weights=weights)
        self.model.conv_proj = nn.Conv2d(
            1,
            self.model.conv_proj.out_channels,
            kernel_size=self.model.conv_proj.kernel_size,
            stride=self.model.conv_proj.stride,
            padding=self.model.conv_proj.padding,
            bias=(self.model.conv_proj.bias is not None),
        )
        self.model.heads.head = nn.Linear(
            self.model.heads.head.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


class ResNet18_sep(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc_right = nn.Linear(in_features, num_classes)
        self.fc_left = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.model(x)
        right_logits = self.fc_right(feat)
        left_logits = self.fc_left(feat)
        return right_logits, left_logits


class ResNet50_sep(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc_right = nn.Linear(in_features, num_classes)
        self.fc_left = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.model(x)
        right_logits = self.fc_right(feat)
        left_logits = self.fc_left(feat)
        return right_logits, left_logits


class ResNet101_sep(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        self.model = models.resnet101(weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc_right = nn.Linear(in_features, num_classes)
        self.fc_left = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.model(x)
        right_logits = self.fc_right(feat)
        left_logits = self.fc_left(feat)
        return right_logits, left_logits


class VGG16_sep(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.model = models.vgg16(weights=weights)
        self.model.features[0] = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1
        )
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Identity()
        self.fc_right = nn.Linear(in_features, num_classes)
        self.fc_left = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.model(x)
        right_logits = self.fc_right(feat)
        left_logits = self.fc_left(feat)
        return right_logits, left_logits


class VGG19_sep(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        self.model = models.vgg19(weights=weights)
        self.model.features[0] = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1
        )
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Identity()
        self.fc_right = nn.Linear(in_features, num_classes)
        self.fc_left = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.model(x)
        right_logits = self.fc_right(feat)
        left_logits = self.fc_left(feat)
        return right_logits, left_logits


class GoogleNet_sep(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
        if pretrained:
            self.model = models.googlenet(
                weights=weights,
                aux_logits=True,
                transform_input=False,
                init_weights=False
            )
            self.model.aux_logits = False
            self.model.aux1 = None
            self.model.aux2 = None
        else:
            self.model = models.googlenet(
                weights=None,
                aux_logits=False,
                transform_input=False,
                init_weights=False
            )
        self.model.conv1.conv = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc_right = nn.Linear(in_features, num_classes)
        self.fc_left = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.model(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        right_logits = self.fc_right(feat)
        left_logits = self.fc_left(feat)
        return right_logits, left_logits


class ViTB16_sep(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_16(weights=weights)
        self.model.conv_proj = nn.Conv2d(
            1,
            self.model.conv_proj.out_channels,
            kernel_size=self.model.conv_proj.kernel_size,
            stride=self.model.conv_proj.stride,
            padding=self.model.conv_proj.padding,
            bias=(self.model.conv_proj.bias is not None),
        )
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Identity()
        self.fc_right = nn.Linear(in_features, num_classes)
        self.fc_left = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.model(x)
        right_logits = self.fc_right(feat)
        left_logits = self.fc_left(feat)
        return right_logits, left_logits


class ViTB32_sep(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_32(weights=weights)
        self.model.conv_proj = nn.Conv2d(
            1,
            self.model.conv_proj.out_channels,
            kernel_size=self.model.conv_proj.kernel_size,
            stride=self.model.conv_proj.stride,
            padding=self.model.conv_proj.padding,
            bias=(self.model.conv_proj.bias is not None),
        )
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Identity()
        self.fc_right = nn.Linear(in_features, num_classes)
        self.fc_left = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.model(x)
        right_logits = self.fc_right(feat)
        left_logits = self.fc_left(feat)
        return right_logits, left_logits


class GoogleNet_clinical(nn.Module):
    def __init__(self, num_classes, clinical_dim, pretrained=True):
        super().__init__()
        weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
        if pretrained:
            self.model = models.googlenet(
                weights=weights,
                aux_logits=True,
                transform_input=False,
                init_weights=False
            )
            self.model.aux_logits = False
            self.model.aux1 = None
            self.model.aux2 = None
        else:
            self.model = models.googlenet(
                weights=None,
                aux_logits=False,
                transform_input=False,
                init_weights=False
            )
        self.model.conv1.conv = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        image_feat_dim = self.model.fc.in_features  # 1024
        self.model.fc = nn.Identity()
        self.classifier = nn.Linear(image_feat_dim + clinical_dim, num_classes)

    def forward(self, x_img, x_clinical):
        img_feat = self.model(x_img)
        if isinstance(img_feat, tuple):
            img_feat = img_feat[0]
        fused_feat = torch.cat([img_feat, x_clinical], dim=1)
        logits = self.classifier(fused_feat)
        return logits


def get_model(model_name, num_classes, pretrained=True, clinical_dim=None):
    model_name = model_name.lower()
    if model_name == "resnet18":
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet50":
        return ResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet101":
        return ResNet101(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vgg11":
        return VGG11(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vgg16":
        return VGG16(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vgg19":
        return VGG19(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "googlenet":
        return GoogleNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "googlenet_clinical":
        if clinical_dim is None:
            raise ValueError("clinical_dim must be provided for googlenet_clinical")
        return GoogleNet_clinical(
            num_classes=num_classes,
            clinical_dim=clinical_dim,
            pretrained=pretrained
        )
    elif model_name == "vit_b_16":
        return ViTB16(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vit_b_32":
        return ViTB32(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet18_sep":
        return ResNet18_sep(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet50_sep":
        return ResNet50_sep(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet101_sep":
        return ResNet101_sep(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vgg16_sep":
        return VGG16_sep(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vgg19_sep":
        return VGG19_sep(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "googlenet_sep":
        return GoogleNet_sep(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vit_b_16_sep":
        return ViTB16_sep(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vit_b_32_sep":
        return ViTB32_sep(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
