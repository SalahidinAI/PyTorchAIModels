# nn_app/ui_streamlit/model_handlers.py
from pathlib import Path
import importlib
import random
import logging

logger = logging.getLogger(__name__)

def _has_module(name):
    return importlib.util.find_spec(name) is not None

# Try to import torch lazily
if _has_module("torch"):
    import torch
else:
    torch = None

def repo_root():
    # repo root is two parents above this file (nn_app/ui_streamlit -> repo_root)
    return Path(__file__).resolve().parents[2]

def load_labels_if_exists(label_filename):
    """Попытаться загрузить файл меток (.pth или .pt). Вернём список строк или None."""
    if torch is None:
        return None
    candidate = repo_root() / label_filename
    if not candidate.exists():
        return None
    try:
        obj = torch.load(candidate, map_location="cpu")
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            # возможный вариант: {'labels': [...]}
            if "labels" in obj and isinstance(obj["labels"], (list, tuple)):
                return [str(x) for x in obj["labels"]]
            # или state_dict-like -> not labels
            #fallthrough
        # last resort: try converting to list
        return [str(x) for x in obj]
    except Exception as e:
        logger.exception("Can't load labels: %s", e)
        return None

def try_load_model(model_filename):
    """Попытаться загрузить torch-модель. Если не удаётся — вернуть None и текст ошибки."""
    if torch is None:
        return None, "torch not installed"
    model_path = repo_root() / model_filename
    if not model_path.exists():
        return None, f"{model_filename} not found in repo root"
    try:
        obj = torch.load(model_path, map_location="cpu")
        # если это полноценный объект nn.Module — хорошо
        if hasattr(obj, "eval"):
            obj.eval()
            return obj, None
        # если это state_dict — вернуть state_dict (пользователь должен сам инстанцировать модель)
        if isinstance(obj, dict):
            # detect if keys look like state_dict keys
            keys = list(obj.keys())
            if keys and isinstance(keys[0], str):
                return {"state_dict": obj}, None
        # иначе — возвращаем объект как есть
        return obj, None
    except Exception as e:
        logger.exception("load_model error")
        return None, str(e)

def mock_predict_from_labels(labels):
    """Вернёт случайный класс + confidence"""
    if not labels:
        labels = [f"class_{i}" for i in range(5)]
    idx = random.randrange(len(labels))
    return {"label": labels[idx], "score": round(random.random() * 0.6 + 0.4, 2)}

# --- API, используемый в app.py ---
def prepare_model_for_inference(model_filename, label_filename=None):
    """
    Попытаться загрузить модель и метки.
    Возврат: dict { 'model': model_or_None, 'labels': list_or_None, 'error': str_or_None }
    """
    labels = None
    if label_filename:
        labels = load_labels_if_exists(label_filename)
    model, err = try_load_model(model_filename)
    return {"model": model, "labels": labels, "error": err}

def safe_predict(model_info, input_tensor=None, topk=3):
    """
    Универсальная safe_predict: если модель есть и выглядит как nn.Module, попробуем вызвать.
    В противном случае вернём mock prediction.
    Возвращает список результатов [{'label':..., 'score':...}, ...]
    """
    # try real predict
    model = model_info.get("model")
    labels = model_info.get("labels")
    if model is None:
        # fallback mock
        res = mock_predict_from_labels(labels)
        return [res]
    # if it's an nn.Module
    try:
        if torch is None:
            return [ {"label": "torch not installed", "score": 0.0} ]
        import torch.nn.functional as F
        with torch.no_grad():
            # If user passed a tensor: try to forward
            if hasattr(model, "eval"):
                model.eval()
                if input_tensor is None:
                    out = model(torch.zeros((1, 3, 64, 64)))  # dummy
                else:
                    # ensure batch dim
                    t = input_tensor
                    if t.ndim == 3:
                        t = t.unsqueeze(0)
                    out = model(t)
                if out.ndim == 1:
                    out = out.unsqueeze(0)
                probs = F.softmax(out, dim=1)
                topk_vals, topk_idx = probs.topk(min(topk, probs.shape[1]), dim=1)
                results = []
                for idx, val in zip(topk_idx[0].tolist(), topk_vals[0].tolist()):
                    label = (labels[idx] if labels and idx < len(labels) else f"class_{idx}")
                    results.append({"label": label, "score": round(float(val), 3)})
                return results
            else:
                # model isn't nn.Module — not callable
                return [ {"label": "model not callable", "score": 0.0} ]
    except Exception as e:
        logger.exception("predict error")
        return [ {"label": "predict failed", "score": 0.0, "error": str(e)} ]
