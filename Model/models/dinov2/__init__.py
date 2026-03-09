from .build_dinov2 import (
    build_dinov2_vits14,
    build_dinov2_vitb14,
    build_dinov2_vitl14,
    build_dinov2_vitg14,
)

dinov2_model_registry = {
    "dinov2_vits14": build_dinov2_vits14,
    "dinov2_vitb14": build_dinov2_vitb14,
    "dinov2_vitl14": build_dinov2_vitl14,
    "dinov2_vitg14": build_dinov2_vitg14,
}
