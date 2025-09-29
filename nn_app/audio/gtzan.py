from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
import torch
import torch.nn as nn
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf
from sqlalchemy.orm import Session
from nn_app.db.database import SessionLocal
from nn_app.db.models import Gtzan


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CheckAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sr = 22050
transform = transforms.MelSpectrogram(
    sample_rate=sr,
    n_mels=64
)

genres = torch.load('labels_gtzan.pth')

model = CheckAudio()
model.load_state_dict((torch.load('model_gtzan.pth', map_location=device)))
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


music_genre_app = APIRouter(prefix='/gtzan', tags=['GTZAN'])


@music_genre_app.post('/predict/')
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Empty file')
        waveform, sample_rate = sf.read(io.BytesIO(data), dtype='float32')
        waveform = torch.tensor(waveform).T

        spec = change_audio(waveform, sample_rate).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_idx = torch.argmax(y_pred, dim=1).item()
            predicted_class = genres[pred_idx]

        gtzan_db = Gtzan(
            audio=file.filename,
            label=predicted_class
        )

        db.add(gtzan_db)
        db.commit()
        db.refresh(gtzan_db)

        return {f'Index: {pred_idx}, Genre: {predicted_class}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error: {e}')
