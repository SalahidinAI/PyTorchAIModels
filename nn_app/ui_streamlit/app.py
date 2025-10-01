import sys
from pathlib import Path

# добавить корень проекта в sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))



# nn_app/ui_streamlit/app.py
import streamlit as st
from pathlib import Path
import io
from PIL import Image
import numpy as np
import os

# local handlers
import nn_app.ui_streamlit.model_handlers as mh

# Apply styles
st.set_page_config(page_title="PyTroch Models — Demo UI", layout="wide", initial_sidebar_state="expanded")

# try to load css
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# repo info
repo_root = Path(__file__).resolve().parents[2]

# Models (filenames that are present in your repo root)
IMAGE_MODELS = ["model_mnist.pth", "model_cifar_100.pth", "model_cnn_fashion_mnist.pth"]
AUDIO_MODELS = ["model_urban.pth", "model_speech.pth", "model_gtzan.pth"]
TEXT_MODELS = ["model_news.pth"]

AUDIO_LABEL_MAP = {
    "model_urban.pth": "labels_urban.pth",
    "model_speech.pth": "labels_speech.pth",
    "model_gtzan.pth": "labels_gtzan.pth",
}

st.sidebar.title("PyTroch UI")
st.sidebar.markdown("Выберите категорию модели и загрузите входные данные. Всё новое — в `nn_app/ui_streamlit/`.")

category = st.sidebar.selectbox("Категория", ["Image", "Audio", "Text"])

if category == "Image":
    chosen = st.sidebar.selectbox("Model (image)", IMAGE_MODELS)
    uploader = st.file_uploader("Upload image (jpg/png)", type=["png", "jpg", "jpeg"])
elif category == "Audio":
    chosen = st.sidebar.selectbox("Model (audio)", AUDIO_MODELS)
    uploader = st.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3", "m4a", "ogg"])
else:
    chosen = st.sidebar.selectbox("Model (text)", TEXT_MODELS)
    uploader = None

st.header("")
col1, col2 = st.columns([3, 2])
with col1:
    # header card
    st.markdown("""
    <div class="header card">
      <div class="logo">AI</div>
      <div>
        <h2 style="margin:0">PyTroch Models — Streamlit UI</h2>
        <div class="small-muted">7 models (image/audio/text). All new UI code lives in <code>nn_app/ui_streamlit/</code></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Quick actions")
    st.button("Reload models (refresh page to fully reload)")
    st.markdown("**Repo root:** `%s`" % repo_root)

st.markdown("---")

# Show model file existence
model_file_path = repo_root / chosen
exists = model_file_path.exists()
st.info(f"Selected model: `{chosen}` — exists: {exists}")

# Try prepare model (this can be slow first time)
label_file = AUDIO_LABEL_MAP.get(chosen, None)
model_info = mh.prepare_model_for_inference(chosen, label_file)

if model_info["error"]:
    st.warning(f"Model load warning: {model_info['error']}. The UI will use a safe fallback predictor (mock) so you can test the interface.")

# Input area & run
if category == "Image":
    if uploader:
        try:
            raw = uploader.read()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)
            # simple preprocessing for demo:
            st.write("Preview dims:", image.size)
            import numpy as np
            # convert to tensor-like array (C,H,W) normalized 0..1
            arr = np.array(image.resize((64,64))).astype(np.float32) / 255.0
            # move channel dim
            tensor = np.transpose(arr, (2,0,1))
            # convert to torch if available
            model_input = None
            try:
                import torch
                model_input = torch.tensor(tensor)
            except Exception:
                model_input = None
            if st.button("Run prediction (image)"):
                results = mh.safe_predict(model_info, input_tensor=model_input)
                st.success("Results:")
                for r in results:
                    st.write(f"- **{r.get('label')}**  (score: {r.get('score')})")
        except Exception as e:
            st.error("Can't read image: " + str(e))
    else:
        st.info("Загрузите изображение через uploader (лёгкое demo-преобразование внутри клиента).")

elif category == "Audio":
    if uploader:
        try:
            raw = uploader.read()
            st.audio(raw)
            st.write("File size:", len(raw), "bytes")
            # For robust audio preproc you can use librosa — but we avoid hard dependency here.
            if st.button("Run prediction (audio)"):
                # we don't pass an audio tensor by default — safe_predict will fallback to mock
                results = mh.safe_predict(model_info, input_tensor=None)
                st.success("Results:")
                for r in results:
                    st.write(f"- **{r.get('label')}**  (score: {r.get('score')})")
        except Exception as e:
            st.error("Can't read audio: " + str(e))
    else:
        st.info("Загрузите аудио файл (wav/mp3). Для реального inference — добавьте preprocessing с librosa/torchaudio.")

else:  # Text
    text = st.text_area("Paste text to classify", height=200)
    if st.button("Run prediction (text)"):
        # For demo: we run safe_predict without tensor
        results = mh.safe_predict(model_info, input_tensor=None)
        st.success("Results:")
        for r in results:
            st.write(f"- **{r.get('label')}**  (score: {r.get('score')})")

st.markdown("---")
st.caption("Если хотите — я могу помочь подключить конкретную функцию infer внутри `model_handlers.py` для каждой модели (image/audio/text) — скажите, какие файлы с architecture/loader у вас есть внутри nn_app, и я автоматически вставлю код.")

