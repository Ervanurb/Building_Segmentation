"""Model bulma/yukleme: YOLO, SamGeo3 (SAM3), Shadow U-Net ve fine-tune
edilmis SAM3 checkpoint'i. Ayrica YOLO egitim metriklerini okuma."""
import glob
import os
import sys
import types
import warnings
import importlib.abc
import importlib.util

import pandas as pd
import streamlit as st
import torch
import segmentation_models_pytorch as smp
from transformers import Sam3Processor, Sam3Model
from ultralytics import YOLO
from samgeo import SamGeo3

from .config import RUNS_DIR


def find_latest_yolo_path():
    weight_files = glob.glob(os.path.join(RUNS_DIR, "yolov11m_geosam_model*", "weights", "best.pt"))
    if not weight_files:
        return None
    return max(weight_files, key=os.path.getmtime)


def load_yolo_metrics(runs_dir=RUNS_DIR):
    csv_files = glob.glob(os.path.join(runs_dir, "yolov11m_geosam_model*", "results.csv"))
    if not csv_files:
        return None
    latest_csv = max(csv_files, key=os.path.getmtime)
    df = pd.read_csv(latest_csv)
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_models(yolo_path):
    yolo_model = YOLO(yolo_path)
    sam = SamGeo3(backend="meta", enable_inst_interactivity=True, confidence_threshold=0.3)
    return yolo_model, sam


def _build_smp_unet(encoder="resnet34"):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


class _LegacyModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, target_names):
        self.target_names = set(target_names)

    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".")[0]
        if top in self.target_names:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)

        def _mod_getattr(name):
            return type(name, (torch.nn.Module,), {})

        mod.__getattr__ = _mod_getattr
        return mod

    def exec_module(self, module):
        pass


def _load_checkpoint_state_dict(checkpoint_path, device):
    try:
        obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except ModuleNotFoundError as e:
        missing_name = str(e).split("'")[1] if "'" in str(e) else "model"
        finder = _LegacyModuleFinder([missing_name])
        sys.meta_path.insert(0, finder)
        try:
            obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
        finally:
            sys.meta_path.remove(finder)
        st.sidebar.info(
            f"Checkpoint eski formatta kaydedilmiş (eksik modül: '{missing_name}'). "
            f"Ağırlıklar uyumluluk modunda kurtarıldı."
        )

    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()

    if isinstance(obj, dict):
        looks_like_state_dict = all(torch.is_tensor(v) for v in obj.values()) if obj else False
        if looks_like_state_dict:
            return obj

        for key in ("model", "model_state_dict", "state_dict", "net", "shadow_unet"):
            if key in obj and isinstance(obj[key], dict):
                extra_info = []
                if "epoch" in obj:
                    extra_info.append(f"epoch={obj['epoch']}")
                if "best_val_iou" in obj:
                    extra_info.append(f"best_val_iou={obj['best_val_iou']:.4f}")
                if extra_info:
                    st.sidebar.info(f"ℹ️ Shadow U-Net checkpoint: {', '.join(extra_info)}")
                return obj[key]

        raise TypeError(
            f"Checkpoint dict ama beklenen anahtarlardan hiçbiri yok. "
            f"Mevcut anahtarlar: {list(obj.keys())}"
        )

    raise TypeError(f"Beklenmeyen checkpoint içeriği: {type(obj)}")


def _load_state_dict_into(model, checkpoint_path, device):
    state = _load_checkpoint_state_dict(checkpoint_path, device)

    model_keys = set(model.state_dict().keys())
    if not (set(state.keys()) & model_keys):
        stripped = {k.split(".", 1)[1]: v for k, v in state.items() if "." in k}
        if set(stripped.keys()) & model_keys:
            state = stripped

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        warnings.warn(
            f"Checkpoint kısmi eşleşti: {len(missing)} eksik, {len(unexpected)} beklenmeyen anahtar"
        )
        st.sidebar.warning(
            f"Checkpoint kısmi eşleşti (eksik: {len(missing)}, beklenmeyen: {len(unexpected)}). "
            f"Encoder adı ('resnet34') checkpoint ile uyuşmuyor olabilir."
        )
    return model


@st.cache_resource
def load_shadow_model(checkpoint_path, encoder="resnet34"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_smp_unet(encoder=encoder)
    model = _load_state_dict_into(model, checkpoint_path, device)
    model = model.to(device).float()
    model.eval()
    return model, device


@st.cache_resource
def load_finetuned_sam3(checkpoint_path):
    """Fine-tune edilmiş SAM3'ü (transformers tabanlı) yükler.
    st.cache_resource sayesinde toggle kapalıyken hiç çağrılmaz;
    açıldığında ise sadece İLK seferde yüklenip belleğe alınır."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.eval()
    if missing or unexpected:
        st.sidebar.warning(
            f"Fine-tune checkpoint kısmi eşleşti (eksik: {len(missing)}, "
            f"beklenmeyen: {len(unexpected)}). Mimari checkpoint ile tam örtüşmüyor olabilir."
        )
    return model, processor, device

