from fastapi import HTTPException, File, UploadFile, APIRouter, Depends
import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from nn_app.db.database import SessionLocal
from nn_app.db.models import Fashion
from sqlalchemy.orm import Session
from nn_app.config import device
import streamlit as st
from PIL import ImageOps


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    # transforms.Resize((28, 28)),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

check_image_app = APIRouter(prefix='/fashion', tags=['Fashion'])
model = CheckImage()
model.load_state_dict(torch.load('model_cnn_fashion_mnist.pth', map_location=device))
model.to(device)
model.eval()

classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',
]


def fashion_image():
    st.title('Fashion MNIST')
    st.text('Upload image with clothes. Model can classify these objects T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.')

    file = st.file_uploader('Choose of drop an image', type=['svg', 'png', 'jpg', 'jpeg'])

    if not file:
        st.warning('No file is uploaded')
    else:
        st.image(file, caption='Uploaded image')
        if st.button('Recognize the image'):
            try:
                image_data = file.read()

                img = Image.open(io.BytesIO(image_data)).convert('L')
                img = ImageOps.invert(img)
                img_tensor = transform(img).unsqueeze(0).to(device)
                print(img_tensor.shape)
                with torch.no_grad():
                    y_pred = model(img_tensor)
                    pred = y_pred.argmax(dim=1).item()

                st.success(f'Prediction: {classes[pred]}')

            except Exception as e:
                st.exception(f'Error: {e}')
