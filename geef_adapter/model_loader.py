from __future__ import annotations

import torch
from omegaconf import OmegaConf

from transfer_model import TransferModel

from .types import ThermoMPNNConfig


def _default_inference_config() -> dict[str, dict[str, object]]:
    return {
        "training": {
            "num_workers": 8,
            "learn_rate": 0.001,
            "epochs": 100,
            "lr_schedule": True,
        },
        "model": {
            "hidden_dims": [64, 32],
            "subtract_mut": True,
            "num_final_layers": 2,
            "freeze_weights": True,
            "load_pretrained": True,
            "lightattn": True,
            "lr_schedule": True,
        },
    }


def resolve_device(device: str) -> str:
    requested = device.strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for ThermoMPNN but torch.cuda.is_available() is False")
    return requested


def _normalize_accel(device: str) -> str:
    return "gpu" if device.startswith("cuda") else "cpu"


def load_runtime_config(local_yaml_path: str, device: str):
    base_cfg = OmegaConf.load(local_yaml_path)
    runtime_cfg = OmegaConf.merge(_default_inference_config(), base_cfg)
    if "platform" in runtime_cfg:
        runtime_cfg.platform.accel = _normalize_accel(device)
    return runtime_cfg


def load_thermompnn_model(config: ThermoMPNNConfig):
    device = resolve_device(config.device)
    runtime_cfg = load_runtime_config(config.local_yaml_path, device)
    model = TransferModel(runtime_cfg)
    checkpoint = torch.load(config.model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {
        key.replace("model.", "", 1) if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.eval()
    model = model.to(device)
    return model
