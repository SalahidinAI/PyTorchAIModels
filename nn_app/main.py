from fastapi import FastAPI
import uvicorn
from nn_app.image import mnist, fashion_mnist, cifar_100
from nn_app.audio import speech_commands, urban, gtzan
from nn_app.text import news


ai_app = FastAPI(title='AI models')

# image
ai_app.include_router(mnist.check_image_app)
ai_app.include_router(fashion_mnist.check_image_app)
ai_app.include_router(cifar_100.check_image_app)

# audio
ai_app.include_router(speech_commands.check_audio)
ai_app.include_router(urban.urban_app)
ai_app.include_router(gtzan.music_genre_app)

# text
ai_app.include_router(news.text_news_app)

if __name__ == '__main__':
    uvicorn.run(ai_app, host='127.0.0.1', port=8000)
