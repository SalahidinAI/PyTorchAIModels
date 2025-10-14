import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
import uvicorn
import streamlit as st

# images
from nn_app.image.mnist import mnist_image
from nn_app.image.fashion_mnist import fashion_image
from nn_app.image.cifar_100 import cifar_100_image

# audio
from nn_app.audio.urban import urban_audio
from nn_app.audio.gtzan import gtzan_audio
from nn_app.audio.speech_commands import speech_audio

# text
from nn_app.text.news import news_text

ai_app = FastAPI(title='AI models')

st.title('AI MODELS')

with st.sidebar:
    st.header('AI Models')
    v = st.radio('Choose', ['MNIST', 'Fashion MNIST', 'CIFAR-100',
                            'Urban', 'GTZAN', 'Speech Commands',
                            'News'])

# images
if v == 'MNIST':
    mnist_image()

elif v == 'Fashion MNIST':
    fashion_image()

elif v == 'CIFAR-100':
    cifar_100_image()

# audio
elif v == 'Urban':
    urban_audio()

elif v == 'GTZAN':
    gtzan_audio()

elif v == 'Speech Commands':
    speech_audio()

# text
elif v == 'News':
    news_text()
