'''ELLA Voice Assistant

This module implements a simple voice assistant named ELLA. It listens for audio
through a USB microphone, transcribes the speech using Vosk, and responds via
text-to-speech using ChatGPT. Speak a phrase beginning with the activation word
followed by your query and ELLA will respond.
'''

import json
import os
import queue
import sys
from typing import Optional

import numpy as np  # noqa: F401
import pyttsx3
import sounddevice as sd
import vosk

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False



def speak(text: str, tts_engine: pyttsx3.engine.Engine) -> None:
    tts_engine.say(text)
    tts_engine.runAndWait()



def load_vosk_model(model_dir: str) -> vosk.Model:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f'Vosk model directory {model_dir!r} not found. '
            'Download a model from https://alphacephei.com/vosk/models and '
            'extract it into this directory.'
        )
    return vosk.Model(model_dir)



def listen_and_transcribe(model: vosk.Model, samplerate: int = 16000) -> str:
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time, status):  # type: ignore[override]
        if status:
            print(f'Audio input status: {status}', file=sys.stderr)
        q.put(bytes(indata))

    rec = vosk.KaldiRecognizer(model, samplerate)
    rec.SetWords(True)

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=callback,
    ):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                result_dict = json.loads(result)
                text = result_dict.get('text', '').strip()
                if text:
                    return text.lower()
            # continue collecting audio if partial result
    return ''



def query_chatgpt(prompt: str) -> str:
    if not _OPENAI_AVAILABLE:
        return 'The ChatGPT API is not available in this environment.'
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return 'No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.'
    openai.api_key = api_key  # type: ignore[attr-defined]
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'You are ELLA, a helpful voice assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.5,
        )
    except Exception as exc:  # noqa: BLE001
        return f'Error contacting ChatGPT: {exc}'
    try:
        return response['choices'][0]['message']['content'].strip()  # type: ignore[index]
    except Exception:
        return 'I received an unexpected response from ChatGPT.'


def main() -> None:
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'vosk-en')
    try:
        model = load_vosk_model(model_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        print('Please download a Vosk model and place it in the specified directory.')
        return
    tts_engine = pyttsx3.init()
    speak('Hello, I am ELLA. Say the activation word followed by your question and I will respond.', tts_engine)
    while True:
        print('Listening... (say the activation word to ask a question)')
        text = listen_and_transcribe(model)
        if not text:
            continue
        if not text.startswith('ella'):
            continue
        query = text[len('ella'):].strip().lstrip(',.:;!')
        if not query:
            speak('Yes?', tts_engine)
            continue
        print(f'User: {query}')
        speak('Thinking...', tts_engine)
        answer = query_chatgpt(query)
        print(f'ELLA: {answer}')
        speak(answer, tts_engine)


if __name__ == '__main__':
    main()
