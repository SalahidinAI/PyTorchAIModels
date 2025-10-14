from fastapi import HTTPException, File, UploadFile, APIRouter, Depends
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from sqlalchemy.orm import Session
from nn_app.db.database import SessionLocal
from nn_app.db.models import Cifar100
from nn_app.config import device
import streamlit as st


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


classes = {
    0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed',
    6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy',
    12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle',
    18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock',
    23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'crab', 27: 'crocodile',
    28: 'cup', 29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish',
    33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house',
    38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower', 42: 'leopard',
    43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree',
    48: 'motorcycle', 49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree',
    53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear',
    58: 'pickup_truck', 59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy',
    63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray',
    68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark',
    74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake',
    79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflower',
    83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television',
    88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle',
    94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'
}

check_image_app = APIRouter(prefix='/cifar_100', tags=['CIFAR 100'])

model_resnet = models.resnet18(weights=None)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 100)
model_resnet.load_state_dict(torch.load("model_cifar_100.pth", map_location=device))
model_resnet = model_resnet.to(device)
model_resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def cifar_100_image():
    st.title('CIFAR-100')
    st.text('Upload image with a number, and model will recognize it')

    file = st.file_uploader('Choose of drop an image', type=['svg', 'png', 'jpg', 'jpeg'])

    if not file:
        st.warning('No file is uploaded')
    else:
        st.image(file, caption='Uploaded image')
        if st.button('Recognize the image'):
            try:
                image_data = file.read()

                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model_resnet(img_tensor)
                    pred = y_pred.argmax(dim=1).item()

                st.success(f'Prediction: {classes[pred]}')

            except Exception as e:
                st.exception(f"Error: {e}")
