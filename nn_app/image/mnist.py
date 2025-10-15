import streamlit as st
import torch
import io
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from sqlalchemy.orm import Session
from nn_app.db.database import SessionLocal
from nn_app.db.models import Mnist
from nn_app.config import device
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from PIL import ImageOps
import numpy as np

# 🟩 NEW: импорт для рисования
from streamlit_drawable_canvas import st_canvas


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

check_image_app = APIRouter(prefix='/mnist', tags=['MNIST'])
model = CheckImage()
model.load_state_dict(torch.load('model_mnist.pth', map_location=device))
model.to(device)
model.eval()


def mnist_image():
    st.title('MNIST Classifier')
    st.text('Upload or draw a digit from 0 to 9, and the model will recognize it.')

    # 🔹 Переключатель: загрузить или нарисовать
    mode = st.radio("Choose input method:", ["Upload", "Draw"], horizontal=True)

    if mode == "Upload":
        # --- Блок загрузки изображения ---
        file = st.file_uploader('Choose or drop an image', type=['svg', 'png', 'jpg', 'jpeg'])

        if not file:
            st.warning('No file is uploaded')
        else:
            st.image(file, caption='Uploaded image')
            if st.button('Recognize the image'):
                try:
                    image_data = file.read()
                    if not image_data:
                        raise HTTPException(status_code=400, detail='No image is given')

                    img = Image.open(io.BytesIO(image_data)).convert('L')  # grayscale
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        y_pred = model(img_tensor)
                        pred = y_pred.argmax(dim=1).item()

                    st.success(f'Prediction: {pred}')

                except Exception as e:
                    st.exception(f'Error: {e}')

    else:
        st.write("Draw a digit below 👇")

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas"
        )

        if canvas_result.image_data is not None:
            if st.button("Recognize the drawn digit"):
                try:
                    arr = canvas_result.image_data.copy()

                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype("uint8")
                    else:
                        arr = arr.astype("uint8")

                    if arr.shape[2] == 4:
                        rgb = arr[:, :, :3].astype("float32")
                        alpha = arr[:, :, 3].astype("float32") / 255.0
                        rgb = (rgb * alpha[..., None])
                        arr = np.clip(rgb, 0, 255).astype("uint8")

                    img = Image.fromarray(arr).convert("L")

                    # 🔹 исправлено здесь
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)

                    arr_l = np.array(img)
                    h, w = arr_l.shape
                    border = np.concatenate([
                        arr_l[:4, :].ravel(),
                        arr_l[-4:, :].ravel(),
                        arr_l[:, :4].ravel(),
                        arr_l[:, -4:].ravel()
                    ])
                    if border.mean() > 127:
                        img = ImageOps.invert(img)

                    # st.image(img.resize((140, 140)), caption="Prepared image for model")

                    to_tensor = transforms.ToTensor()
                    img_tensor = to_tensor(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        y_pred = model(img_tensor)
                        pred = y_pred.argmax(dim=1).item()

                    st.success(f'Prediction: {pred}')

                except Exception as e:
                    st.exception(f'Error: {e}')
