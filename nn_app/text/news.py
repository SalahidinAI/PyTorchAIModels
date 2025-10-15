# add google translate
import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from torchtext.data import get_tokenizer
from googletrans import Translator
from sqlalchemy.orm import Session
from nn_app.db.database import SessionLocal
from nn_app.db.schema import TextSchema
from nn_app.db.models import News
from nn_app.config import device
import streamlit as st


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


classes = ['World', 'Sports', 'Business', 'Sci/Tech']


class CheckNews(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.lin = nn.Linear(128, 4)

    def forward(self, x):
        x = self.emb(x)
        _, (x, _) = self.lstm(x)
        x = self.lin(x.squeeze(0))
        return x


vocab = torch.load('vocab_news.pth', weights_only=False)
tokenizer = get_tokenizer('basic_english')


def text_pipeline(text: str):
    return [vocab[i] for i in tokenizer(text)]


model = CheckNews(len(vocab)).to(device)
model.load_state_dict(torch.load('model_news.pth', map_location=device))
model.eval()

text_news_app = APIRouter(prefix='/news', tags=['News'])

translator = Translator()


def news_text():
    st.title('News Classifier')
    st.text(f'Type a text and model will recognize it. Classes {vocab}')

    text = st.text_input('Type')

    if not text:
        st.warning('No text typed')
    else:
        if st.button('Classify Text'):
            translated = translator.translate(text, dest='en').text
            translated_text = translated

            tensor = torch.tensor(text_pipeline(translated_text), dtype=torch.int64).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor)
                label = torch.argmax(pred, dim=1).item()

            st.success(f'Label: {classes[label]}')
