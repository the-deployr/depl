from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import asyncio
import threading
import base64
import google.generativeai as genai
from google.generativeai import types

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

MODEL = "models/gemini-2.0-flash-live-001"
GEMINI_API_KEY = "AIzaSyAOCk8-5OSa-J0T0o4PhRsc6qT7-ttCcc4"  # Replace securely

LANGUAGES = {
    'English': {'code': 'en-US'}, 'Spanish': {'code': 'es-ES'}, 'French': {'code': 'fr-FR'},
    'German': {'code': 'de-DE'}, 'Italian': {'code': 'it-IT'}, 'Portuguese': {'code': 'pt-BR'},
    'Hindi': {'code': 'hi-IN'}, 'Chinese': {'code': 'zh-CN'}, 'Japanese': {'code': 'ja-JP'},
    'Korean': {'code': 'ko-KR'}, 'Arabic': {'code': 'ar-SA'}, 'Russian': {'code': 'ru-RU'}, 'Tamil': {'code': 'ta-IN'}
}
VOICES = {'male': 'Fenrir', 'female': 'Leda'}

genai.configure(api_key=GEMINI_API_KEY)

class WebSpeechTranslator:
    def __init__(self, target_language, gender, socket_id):
        self.target_language = target_language
        self.gender = gender
        self.socket_id = socket_id
        self.audio_queue = asyncio.Queue()
        self.is_running = False
        self.session = None

    def get_config(self):
        lang_config = LANGUAGES.get(self.target_language, LANGUAGES['English'])
        voice_name = VOICES.get(self.gender, 'Leda')
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                language_code=lang_config['code'],
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(f"You are a speech translator that converts audio to spoken {self.target_language}...")],
                role="user"
            ),
        )

    async def send_audio(self):
        try:
            while self.is_running:
                audio_data = await self.audio_queue.get()
                await self.session.send(input=audio_data)
        except Exception as e:
            socketio.emit('error', {'message': f'Send error: {str(e)}'}, room=self.socket_id)

    async def receive_audio(self):
        try:
            while self.is_running:
                turn = self.session.receive()
                async for response in turn:
                    if response.data:
                        continue  # ignoring streamed audio playback for now
                    if response.text:
                        socketio.emit('translation', {
                            'text': response.text,
                            'language': self.target_language
                        }, room=self.socket_id)
        except Exception as e:
            socketio.emit('error', {'message': f'Receive error: {str(e)}'}, room=self.socket_id)

    async def run(self):
        try:
            async with genai.aio.live.connect(model=MODEL, config=self.get_config()) as session:
                self.session = session
                self.is_running = True

                socketio.emit('status', {'message': f'Translation started for {self.target_language}'}, room=self.socket_id)

                await asyncio.gather(
                    self.send_audio(),
                    self.receive_audio(),
                    return_exceptions=True
                )
        except Exception as e:
            socketio.emit('error', {'message': f'Session error: {str(e)}'}, room=self.socket_id)
        finally:
            self.is_running = False

active_translators = {}

@app.route('/')
def index():
    return render_template('index.html', languages=list(LANGUAGES.keys()))

@socketio.on('connect')
def on_connect():
    emit('status', {'message': 'Connected to server. Select language and voice to start.'})

@socketio.on('disconnect')
def on_disconnect():
    if request.sid in active_translators:
        active_translators[request.sid].is_running = False
        del active_translators[request.sid]

@socketio.on('start_translation')
def start_translation(data):
    lang = data.get('language')
    gender = data.get('gender')
    if lang not in LANGUAGES or gender not in VOICES:
        emit('error', {'message': 'Invalid language or voice gender'})
        return

    if request.sid in active_translators:
        active_translators[request.sid].is_running = False

    translator = WebSpeechTranslator(lang, gender, request.sid)
    active_translators[request.sid] = translator

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(translator.run())

    threading.Thread(target=runner, daemon=True).start()

@socketio.on('audio_chunk')
def receive_chunk(data):
    if request.sid not in active_translators:
        emit('error', {'message': 'No active translation session'})
        return
    try:
        audio = base64.b64decode(data['audio'])
        active_translators[request.sid].audio_queue.put_nowait({"data": audio, "mime_type": "audio/webm"})
    except Exception as e:
        emit('error', {'message': f'Audio decode error: {str(e)}'})

@socketio.on('stop_translation')
def stop_translation():
    if request.sid in active_translators:
        active_translators[request.sid].is_running = False
        del active_translators[request.sid]
        emit('status', {'message': 'Translation stopped'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)