# streamlit_ui/app.py
import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pkgutil, importlib, inspect

# Путь к корню репозитория (одна папка выше streamlit_ui)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT

st.set_page_config(page_title="PyTroch — Streamlit UI", layout="wide")

def inject_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def discover_model_files():
    files = list(MODEL_DIR.glob("*.pth")) + list(MODEL_DIR.glob("*.pt"))
    files = sorted(files, key=lambda p: p.name.lower())
    return files

def guess_model_type(name):
    n = name.lower()
    if any(x in n for x in ("mnist","fashion","cifar","cifar100")):
        return "image"
    if any(x in n for x in ("gtzan","speech","urban","audio")):
        return "audio"
    if any(x in n for x in ("news","vocab","text")):
        return "text"
    return "unknown"

def try_load_torchscript(path):
    try:
        m = torch.jit.load(str(path), map_location="cpu")
        return m
    except Exception:
        return None

def load_checkpoint(path: Path):
    """
    Попытка загрузить .pth:
      - torch.jit (если это скриптовая модель)
      - если сохранён полный nn.Module (pickle) — вернуть его
      - если state_dict — попытаться найти класс в nn_app и восстановить
    """
    device = torch.device("cpu")
    ts = try_load_torchscript(path)
    if ts is not None:
        return ts
    ckpt = torch.load(str(path), map_location=device)
    if isinstance(ckpt, nn.Module):
        return ckpt
    if isinstance(ckpt, dict):
        # возможные ключи
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt  # возможно уже state_dict
        # попробуем найти класс в пакете nn_app
        try:
            import nn_app
            candidates = {}
            for finder, name, ispkg in pkgutil.iter_modules(nn_app.__path__):
                mod = importlib.import_module(f"nn_app.{name}")
                for cname, cls in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(cls, nn.Module):
                        candidates[cname.lower()] = cls
            # ищем лучший матч по имени файла
            stem = path.stem.lower()
            best_cls = None
            for cname, cls in candidates.items():
                if cname in stem:
                    best_cls = cls
                    break
            if best_cls is None and candidates:
                best_cls = list(candidates.values())[0]
            if best_cls is not None:
                model = best_cls()
                model.load_state_dict(state)
                model.eval()
                return model
        except Exception as e:
            raise RuntimeError(f"Не удалось автоматически восстановить модель из state_dict: {e}\nВам может потребоваться импорт класса модели из nn_app или пересохранить модель.") from e
    raise RuntimeError("Неподдерживаемый формат .pth (не torchscript, не pickled Module, не state-dict)")

@st.cache_resource(show_spinner=False)
def get_model(path_str: str):
    path = Path(path_str)
    return load_checkpoint(path)

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    x = transform(img).unsqueeze(0)
    return x, img

def softmax_tensor(tensor):
    with torch.no_grad():
        exp = torch.exp(tensor - tensor.max())
        return (exp / exp.sum()).cpu().numpy()

def main():
    inject_css()
    st.markdown('<div class="header"><h1>PyTroch — Beautiful Streamlit UI</h1><p class="subtitle">7 models — images, audio, text. Choose model & run inference.</p></div>', unsafe_allow_html=True)

    files = discover_model_files()
    model_map = {}
    for f in files:
        model_map.setdefault(guess_model_type(f.name), []).append(f)

    # Sidebar
    st.sidebar.title("Controls")
    model_type = st.sidebar.selectbox("Model type", ["image", "audio", "text", "unknown"], index=0)
    type_files = model_map.get(model_type, [])
    if not type_files:
        st.sidebar.info("No model files found for this type. Check repo root for .pth files.")
    selected = None
    if type_files:
        selected = st.sidebar.selectbox("Choose model file", type_files, format_func=lambda p: p.name)
    st.sidebar.markdown("---")
    st.sidebar.write("Run on CPU by default. Use models saved in repo root.")

    # UI for each type
    if model_type == "image":
        uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
        if uploaded and selected:
            st.image(uploaded, caption="Input image", use_column_width=False)
            try:
                model = get_model(str(selected))
            except Exception as e:
                st.error(f"Ошибка загрузки модели: {e}")
                return
            x, pil_img = preprocess_image(uploaded)
            try:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                probs = softmax_tensor(out[0])
                top = int(probs.argmax())
                st.success(f"Predicted class index: {top}")
                st.bar_chart(probs)
            except Exception as e:
                st.error(f"Inference failed: {e}")

    elif model_type == "text":
        txt = st.text_area("Enter text to classify", height=200)
        if st.button("Classify text") and txt.strip() and selected:
            try:
                model = get_model(str(selected))
            except Exception as e:
                st.error(f"Ошибка загрузки модели: {e}")
                return
            # try to use saved vocab if exists
            vocab_path = MODEL_DIR / "vocab_news.pth"
            vocab = None
            if vocab_path.exists():
                try:
                    vocab = torch.load(str(vocab_path))
                except Exception:
                    vocab = None
            try:
                if vocab:
                    tokens = txt.lower().split()
                    ids = [vocab.get(t, 0) for t in tokens]
                    x = torch.tensor(ids).unsqueeze(0)
                    out = model(x)
                else:
                    out = model([txt])
                probs = softmax_tensor(out[0])
                st.write("Top predicted index:", int(probs.argmax()))
                st.bar_chart(probs)
            except Exception as e:
                st.error(f"Text inference error: {e}")

    elif model_type == "audio":
        uploaded = st.file_uploader("Upload audio (wav/mp3)", type=["wav","mp3","ogg","flac"])
        if uploaded and selected:
            try:
                import librosa
                data, sr = librosa.load(uploaded, sr=16000, mono=True)
                mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
                x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()
                model = get_model(str(selected))
                out = model(x)
                probs = softmax_tensor(out[0])
                st.write("Top prediction index:", int(probs.argmax()))
                st.bar_chart(probs)
            except Exception as e:
                st.error(f"Audio inference error (install librosa?): {e}")
    else:
        st.write("Выберите тип модели в сайдбаре. Я просканировал корень репозитория на .pth файлы.")

if __name__ == "__main__":
    main()
