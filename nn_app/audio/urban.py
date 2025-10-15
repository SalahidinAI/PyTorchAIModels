# pip install soundfile torchaudio

from fastapi import HTTPException, UploadFile, File, APIRouter, Depends
import torch
import torch.nn as nn
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf
from sqlalchemy.orm import Session
from nn_app.db.database import SessionLocal
from nn_app.db.models import Urban
from nn_app.config import device
import streamlit as st


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class UrbanAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(UrbanAudio, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.flatten = nn.Flatten()

        self.second = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.flatten(x)
        x = self.second(x)
        return x


sr = 22050
transform = transforms.MelSpectrogram(
    sample_rate=sr,
    n_mels=64
)


labels = torch.load('labels_urban.pth')

model = UrbanAudio()
model.load_state_dict((torch.load('model_urban.pth', map_location=device)))
model.to(device)
model.eval()

max_len = 500

def change_audio(waveform, sample_rate):
    if sample_rate != sr:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        # waveform = resample(waveform)
        waveform = resample(torch.tensor(waveform).unsqueeze(0))

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]

    elif spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))

    return spec


urban_app = APIRouter(prefix='/urban', tags=['Urban sounds'])


def urban_audio():
    st.title('Urban Sounds Classifier')
    st.text(f'Upload audio (.wav) to recognize sound. Model can classify sounds of {labels}')

    file = st.file_uploader('Upload a file', type=['wav'])

    if not file:
        st.warning('Upload a file')
    else:
        st.audio(file)
        if st.button('Recognize'):
            try:
                data = file.read()
                if not data:
                    raise HTTPException(status_code=400, detail='Empty file')
                waveform, sample_rate = sf.read(io.BytesIO(data), dtype='float32')
                waveform = torch.tensor(waveform).T

                spec = change_audio(waveform, sample_rate).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model(spec)
                    pred_idx = torch.argmax(y_pred, dim=1).item()
                    predicted_class = labels[pred_idx]

                st.success(f'Index: {pred_idx}, Sound: {predicted_class}')

            except Exception as e:
                st.exception(f'Error: {e}')
