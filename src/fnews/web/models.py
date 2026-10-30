from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision.models import ResNet18_Weights, resnet18
from transformers import CLIPModel, CLIPProcessor

from fnews.experiments.neural_text import encode_documents
from fnews.ml.features.tfidf import ManualTfidfVectorizer
from fnews.ml.label_semantics import (
    fake_label_from_source_prediction,
    fake_margin_from_source_true_margin,
    fake_probability_from_source_true_probability,
)
from fnews.ml.metrics.manual import PlattCalibrator
from fnews.ml.models.clip_fusion import GatedFusionHead
from fnews.ml.models.cnn import ManualCNNClassifier
from fnews.ml.models.linear_svm import ManualLinearSVM
from fnews.ml.models.text_embedding import ManualTextEmbeddingClassifier
from fnews.ml.models.weighted_fusion import weighted_fake_probability
from fnews.ml.preprocessing.text import load_vocabulary, tokenize

ROOT = Path(__file__).resolve().parents[3]
TEXT_DIR = ROOT / "reports/experiments/neural_text_50k_tuned"
SVM_DIR = ROOT / "reports/manual_svm_repair/manual_tfidf_svm_v1"
WEIGHTED_DIR = ROOT / "reports/weighted_late_fusion_v1"
RESNET_CHECKPOINT = ROOT / "reports/experiments/research_dataset_baselines/resnet18_frozen/best_model.pt"
IMAGE_DIR = ROOT / "reports/experiments/image_cnn_pilot/best_model"
FUSION_DIR = ROOT / "reports/multimodal_fusion/clip_fusion_v1/gated"
FUSION_CALIBRATION = ROOT / "reports/phase2/calibration/temperature.json"
MAX_TEXT = 100_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


class ModelRegistry:
    """Lazy, process-local model registry; checkpoints are loaded once per worker."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded: dict[str, Any] = {}
        self._failures: dict[str, str] = {}

    def models(self) -> list[dict[str, Any]]:
        items = [
            self._describe_tfidf_svm(),
            self._describe_text(),
            self._describe_image(),
            self._describe_weighted_fusion(),
            self._describe_fusion(),
            {
                "id": "image-text-mismatch-unavailable",
                "version": "unavailable",
                "task": "image_text_mismatch",
                "modalities": ["text", "image"],
                "architecture": "trained mismatch detector not found",
                "availability": "unavailable",
                "reason": "Fake-news labels do not measure image-text consistency; no mismatch detector is available.",
                "metrics": None,
                "calibrated": False,
            },
        ]
        return items

    def predict(
        self, text: str, image_path: Path | None, selected: list[str]
    ) -> list[dict[str, Any]]:
        output = []
        for model in self.models():
            model_id = model["id"]
            if model_id not in selected:
                continue
            if model["availability"] != "available":
                output.append(
                    {"model_id": model_id, "status": "unavailable", "error": model["reason"]}
                )
                continue
            started = time.perf_counter()
            try:
                if model_id == "fakeddit-tfidf-linearsvc-research-v1":
                    label, margin = self._predict_tfidf_svm(text)
                    output.append({
                        "model_id": model_id, "model_version": "research-v1", "task": "fake_news_classification",
                        "status": "completed", "label": label, "decision_margin_fake": margin,
                        "score_name": "LinearSVC non-true-positive decision margin", "threshold": 0.0,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "device": "CPU", "warning": "Research model trained on Fakeddit; the SVM margin is not a probability or factuality verdict.",
                    })
                    continue
                if model_id == "fakeddit-weighted-late-fusion-v1":
                    if not text.strip() and image_path is None:
                        raise ValueError("Weighted late fusion requires at least one modality.")
                    if not text.strip():
                        image_probability = self._predict_resnet(image_path)
                        output.append({
                            "model_id": model_id, "model_version": "v1", "task": "fake_news_classification",
                            "status": "completed", "label": "non_true" if image_probability >= 0.5 else "true",
                            "raw_score_non_true": image_probability, "raw_score_fake": image_probability,
                            "image_probability_non_true": image_probability,
                            "mode": "image_only_fallback", "score_name": "image-only fallback probability",
                            "threshold": 0.5, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                            "warning": "No text was supplied; this is image-only fallback, not weighted fusion.",
                        })
                        continue
                    _, margin = self._predict_tfidf_svm(text)
                    calibrator = self._load_weighted_calibrator()
                    text_probability = fake_probability_from_source_true_probability(
                        float(calibrator.predict_proba([[-margin]])[0, 1])
                    )
                    if image_path is None:
                        output.append({
                            "model_id": model_id, "model_version": "v1", "task": "fake_news_classification",
                            "status": "completed", "label": "non_true" if text_probability >= 0.5 else "true",
                            "raw_score_non_true": text_probability, "raw_score_fake": text_probability,
                            "text_probability_non_true": text_probability,
                            "mode": "text_only_fallback", "score_name": "calibrated text-only fallback probability",
                            "threshold": 0.5, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                            "warning": "No image was supplied; this is text-only fallback, not weighted fusion.",
                        })
                        continue
                    image_probability = self._predict_resnet(image_path)
                    alpha = float(self._weighted_config()["alpha"])
                    score = weighted_fake_probability(text_probability, image_probability, alpha)
                    output.append({
                        "model_id": model_id, "model_version": "v1", "task": "fake_news_classification",
                        "status": "completed", "label": "non_true" if score >= 0.5 else "true",
                        "raw_score_non_true": score, "raw_score_fake": score,
                        "text_probability_non_true": text_probability,
                        "image_probability_non_true": image_probability, "alpha": alpha,
                        "score_name": "weighted late-fusion non-true probability", "threshold": 0.5,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "device": "MPS" if torch.backends.mps.is_available() else "CPU",
                        "warning": "Optional validation-selected research model; not the production default or a factuality verdict.",
                    })
                    continue
                if model_id == "fakeddit-text-embedding-50k-seed42-v1":
                    score = self._predict_text(text)
                elif model_id == "fakeddit-clip-gated-fusion-v1":
                    if image_path is None or not text.strip():
                        raise ValueError("The fusion model requires both text and an image.")
                    score, fusion_input = self._predict_fusion(text, image_path)
                else:
                    if image_path is None:
                        output.append(
                            {
                                "model_id": model_id,
                                "status": "skipped",
                                "error": "This model requires an image.",
                            }
                        )
                        continue
                    score = self._predict_image(image_path)
                result = {
                        "model_id": model_id,
                        "model_version": "1",
                        "preprocessing_version": "fakeddit-preprocess-v1",
                        "task": "fake_news_classification",
                        "status": "completed",
                        "label": "non_true" if score >= 0.5 else "true",
                        "raw_score_non_true": score, "raw_score_fake": score,
                        "score_name": "uncalibrated Fakeddit non-true score",
                        "threshold": 0.5,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "device": "MPS" if torch.backends.mps.is_available() else "CPU",
                        "warning": "Research model trained on Fakeddit; score is not a calibrated probability or a factuality verdict.",
                    }
                if model_id == "fakeddit-clip-gated-fusion-v1":
                    result["input_diagnostics"] = fusion_input
                    calibrated = self._calibrate_fusion_score(score)
                    if calibrated is not None:
                        result["calibrated_score_non_true"] = calibrated
                        result["calibrated_score_fake"] = calibrated
                        result["calibration_status"] = "validation_half_temperature_scaling_v1"
                output.append(result)
            except Exception as exc:  # Keep successful independent model outputs.
                output.append(
                    {
                        "model_id": model_id,
                        "status": "failed",
                        "error": f"Inference failed: {type(exc).__name__}",
                    }
                )
        return output

    def _describe_fusion(self) -> dict[str, Any]:
        metrics = self._read_json(FUSION_DIR.parent / "metrics.json") or {}
        return {
            "id": "fakeddit-clip-gated-fusion-v1",
            "version": "1",
            "task": "fake_news_classification",
            "modalities": ["text", "image"],
            "architecture": "frozen CLIP ViT-B/32 embeddings + trained gated fusion head",
            "checkpoint": "reports/multimodal_fusion/clip_fusion_v1/gated/best_model.pt",
            "preprocessing": "CLIPProcessor paired text + RGB image",
            "label_mapping": {"0": "non_true/fake taxonomy", "1": "true"},
            "availability": "archived_inactive",
            "reason": "Archived inactive for the research-v1 application rollback; checkpoint preserved.",
            "metrics": self._present_source_metrics(metrics.get("results", {}).get("gated", {}).get("validation")),
            "metrics_source": "fnews-research-v1 validation",
            "calibrated": FUSION_CALIBRATION.is_file(),
            "calibration": "validation-half temperature scaling v1" if FUSION_CALIBRATION.is_file() else None,
        }

    def _describe_tfidf_svm(self) -> dict[str, Any]:
        metrics = self._read_json(ROOT / "reports/experiments/research_dataset_baselines/metrics.json") or {}
        return {
            "id": "fakeddit-tfidf-linearsvc-research-v1", "version": "manual-repair-v1",
            "task": "fake_news_classification", "modalities": ["text"],
            "architecture": "repository ManualTfidfVectorizer + ManualLinearSVM",
            "checkpoint": "reports/manual_svm_repair/manual_tfidf_svm_v1/model",
            "preprocessing": "saved repository-owned TF-IDF vectorizer", "label_mapping": {"0": "non_true/fake taxonomy", "1": "true"},
            "availability": "available" if (SVM_DIR / "model/model.npz").is_file() and (SVM_DIR / "vectorizer/tfidf_arrays.npz").is_file() else "unavailable",
            "reason": None, "metrics": self._present_source_metrics(metrics.get("results", {}).get("tfidf_svm", {}).get("validation")),
            "metrics_source": "manual repair v1: balanced 50k train subset, untouched official validation",
            "calibrated": False,
        }

    def _describe_weighted_fusion(self) -> dict[str, Any]:
        available = (WEIGHTED_DIR / "metrics.json").is_file() and (WEIGHTED_DIR / "svm_platt_calibrator.joblib").is_file() and RESNET_CHECKPOINT.is_file()
        return {
            "id": "fakeddit-weighted-late-fusion-v1", "version": "v1", "task": "fake_news_classification",
            "modalities": ["text", "image"], "architecture": "Platt-calibrated TF-IDF LinearSVC + frozen ResNet-18 softmax; weighted probability sum",
            "checkpoint": "reports/weighted_late_fusion_v1/svm_platt_calibrator.joblib", "preprocessing": "saved SVM pipeline + ResNet18 DEFAULT transforms",
            "label_mapping": {"0": "non_true/fake taxonomy", "1": "true"}, "availability": "available" if available else "unavailable",
            "reason": None if available else "Required calibrated fusion artifacts are missing.",
            "metrics": self._weighted_config().get("validation", {}).get("selected") if available else None,
            "metrics_source": "validation-only alpha selection; not production default", "calibrated": True,
        }

    @staticmethod
    def _calibrate_fusion_score(score: float) -> float | None:
        """Apply saved binary temperature scaling without changing the raw decision."""
        try:
            temperature = float(json.loads(FUSION_CALIBRATION.read_text())["temperature"])
            if not temperature > 0:
                return None
            clipped = min(max(score, 1e-7), 1 - 1e-7)
            return float(1 / (1 + math.exp(-math.log(clipped / (1 - clipped)) / temperature)))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    def _present_source_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
        """Translate historical raw-class metric labels for registry display."""
        if not metrics:
            return metrics
        result = dict(metrics)
        per_class = metrics.get("per_class")
        if isinstance(per_class, dict) and "true" in per_class and "fake" in per_class:
            translated = dict(per_class)
            translated["true"] = per_class["fake"]
            translated["non_true"] = per_class["true"]
            translated.pop("fake", None)
            result["per_class"] = translated
        matrix = metrics.get("confusion_matrix")
        if isinstance(matrix, list) and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix):
            result["confusion_matrix"] = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]
            result["confusion_matrix_order"] = ["true", "non_true"]
        return result

    def _describe_text(self) -> dict[str, Any]:
        metrics = self._read_json(TEXT_DIR / "metrics.json")
        available = all(
            (TEXT_DIR / p).is_file()
            for p in ("model/model.npz", "model/model_metadata.json", "vocabulary.json")
        )
        return {
            "id": "fakeddit-text-embedding-50k-seed42-v1",
            "version": "1",
            "task": "fake_news_classification",
            "modalities": ["text"],
            "architecture": "manual NumPy embedding + masked mean pooling + dense classifier",
            "checkpoint": "reports/experiments/neural_text_50k_tuned/model",
            "preprocessing": "minimal tokenizer; URL normalization; lowercase; max 32 tokens; <PAD>/<UNK> vocabulary",
            "label_mapping": {"0": "non_true/fake taxonomy", "1": "true"},
            "availability": "available" if available else "unavailable",
            "reason": None if available else "Required checkpoint or vocabulary file is missing.",
            "metrics": metrics.get("validation_metrics") if metrics else None,
            "metrics_source": "Fakeddit validation split, 50k sampled training run; reports/experiments/neural_text_50k_tuned/metrics.json",
            "calibrated": False,
        }

    def _describe_image(self) -> dict[str, Any]:
        metrics = self._read_json(ROOT / "reports/experiments/image_cnn_pilot/metrics.json")
        metadata = self._read_json(IMAGE_DIR / "model_metadata.json") or {}
        available = all((IMAGE_DIR / p).is_file() for p in ("model.npz", "model_metadata.json"))
        return {
            "id": "fakeddit-image-cnn-pilot-seed42-v1",
            "version": "1",
            "task": "fake_news_classification",
            "modalities": ["image"],
            "architecture": "manual CNN: 8/16/32 convolution channels, global average pooling, dense classifier",
            "checkpoint": "reports/experiments/image_cnn_pilot/best_model",
            "preprocessing": "RGB, bilinear resize to 64x64, [0,1], recorded training channel mean/std normalization",
            "label_mapping": {"0": "non_true/fake taxonomy", "1": "true"},
            "availability": "available" if available else "unavailable",
            "reason": None if available else "Required checkpoint file is missing.",
            "metrics": metrics.get("validation_metrics") if metrics else None,
            "metrics_source": "Fakeddit image pilot validation sample; reports/experiments/image_cnn_pilot/metrics.json",
            "calibrated": False,
            "preprocessing_metadata": {
                key: metadata.get("metadata", {}).get(key)
                for key in ("image_size", "channel_mean", "channel_std")
            },
        }

    def _predict_text(self, text: str) -> float:
        if not text or len(text) > MAX_TEXT:
            raise ValueError(f"Text must contain 1 to {MAX_TEXT} characters.")
        with self._lock:
            model, vocabulary, max_length = self._load_text()
        tokens = tokenize(text, mode="minimal")
        batch = encode_documents([tokens], vocabulary, max_length)
        return fake_probability_from_source_true_probability(float(model.predict_proba(batch)[0]))

    def _predict_tfidf_svm(self, text: str) -> tuple[str, float]:
        if not text or len(text) > MAX_TEXT:
            raise ValueError(f"Text must contain 1 to {MAX_TEXT} characters.")
        with self._lock:
            if "tfidf_svm" not in self._loaded:
                self._loaded["tfidf_svm"] = ManualLinearSVM.load(SVM_DIR / "model")
                self._loaded["tfidf_vectorizer"] = ManualTfidfVectorizer.load(SVM_DIR / "vectorizer")
            model, vectorizer = self._loaded["tfidf_svm"], self._loaded["tfidf_vectorizer"]
        vector = vectorizer.transform([text])
        source_true_margin = float(model.decision_function(vector)[0])
        source_prediction = int(model.predict(vector)[0])
        return fake_label_from_source_prediction(source_prediction), fake_margin_from_source_true_margin(source_true_margin)

    def _weighted_config(self) -> dict[str, Any]:
        return self._read_json(WEIGHTED_DIR / "metrics.json") or {}

    def _load_weighted_calibrator(self):
        with self._lock:
            if "weighted_calibrator" not in self._loaded:
                self._loaded["weighted_calibrator"] = PlattCalibrator.load(
                    WEIGHTED_DIR / "svm_platt_calibrator.json"
                )
            return self._loaded["weighted_calibrator"]

    def _predict_resnet(self, path: Path) -> float:
        with self._lock:
            if "weighted_resnet" not in self._loaded:
                model = resnet18(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.load_state_dict(torch.load(RESNET_CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"])
                self._loaded["weighted_resnet"] = model.eval()
                self._loaded["weighted_transform"] = ResNet18_Weights.DEFAULT.transforms()
            model, transform = self._loaded["weighted_resnet"], self._loaded["weighted_transform"]
        with Image.open(path) as image:
            pixels = transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return fake_probability_from_source_true_probability(float(torch.softmax(model(pixels), dim=1)[0, 1]))

    def _predict_image(self, path: Path) -> float:
        with self._lock:
            model, mean, std = self._load_image()
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float64).transpose(2, 0, 1) / 255.0
        arr = (arr - mean[:, None, None]) / std[:, None, None]
        return fake_probability_from_source_true_probability(float(model.predict_proba(arr[None, ...])[0]))

    def _predict_fusion(self, text: str, path: Path) -> tuple[float, dict[str, object]]:
        with self._lock:
            if "fusion" not in self._loaded:
                checkpoint = torch.load(
                    FUSION_DIR / "best_model.pt", map_location="cpu", weights_only=False
                )
                head = GatedFusionHead(
                    hidden_dim=int(checkpoint["config"]["hidden_dim"]),
                    dropout=float(checkpoint["config"]["dropout"]),
                )
                head.load_state_dict(checkpoint["state_dict"])
                head.eval()
                self._loaded["fusion"] = head
                self._loaded["fusion_mean"] = torch.tensor(
                    checkpoint["feature_mean"], dtype=torch.float32
                )
                self._loaded["fusion_std"] = torch.tensor(
                    checkpoint["feature_std"], dtype=torch.float32
                )
                self._loaded["clip"] = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).eval()
                self._loaded["clip_processor"] = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            max_length = 64
            token_count = len(self._loaded["clip_processor"].tokenizer(text, truncation=False)["input_ids"])
            batch = self._loaded["clip_processor"](
                text=[text],
                images=[rgb],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                features = torch.cat(
                    (
                        self._loaded["clip"].get_image_features(pixel_values=batch["pixel_values"]),
                        self._loaded["clip"].get_text_features(
                            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                        ),
                    ),
                    dim=1,
                ).float()
            features = (features - self._loaded["fusion_mean"]) / self._loaded["fusion_std"]
            score = fake_probability_from_source_true_probability(
                float(torch.softmax(self._loaded["fusion"](features[:, :512], features[:, 512:]), dim=1)[0, 1].detach())
            )
            return score, {
                "clip_text_max_length": max_length,
                "clip_tokens_before_truncation": token_count,
                "clip_text_truncated": token_count > max_length,
                "image_preprocessing": "PIL RGB + CLIPProcessor",
                "training_embedding_protocol": "CLIPProcessor paired text/image, max_length=64",
            }

    def _load_text(self) -> tuple[ManualTextEmbeddingClassifier, dict[str, int], int]:
        if "text" not in self._loaded:
            self._loaded["text"] = ManualTextEmbeddingClassifier.load(TEXT_DIR / "model")
            self._loaded["text_vocab"] = load_vocabulary(TEXT_DIR / "vocabulary.json")
            meta = self._read_json(TEXT_DIR / "model/model_metadata.json") or {}
            self._loaded["text_max_length"] = int(meta.get("metadata", {}).get("max_length", 32))
        return self._loaded["text"], self._loaded["text_vocab"], self._loaded["text_max_length"]

    def _load_image(self) -> tuple[ManualCNNClassifier, np.ndarray, np.ndarray]:
        if "image" not in self._loaded:
            meta = self._read_json(IMAGE_DIR / "model_metadata.json") or {}
            model = ManualCNNClassifier.load(IMAGE_DIR)
            details = meta.get("metadata", {})
            self._loaded["image"] = model
            self._loaded["image_mean"] = np.asarray(details["channel_mean"], dtype=np.float64)
            self._loaded["image_std"] = np.asarray(details["channel_std"], dtype=np.float64)
        return self._loaded["image"], self._loaded["image_mean"], self._loaded["image_std"]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
