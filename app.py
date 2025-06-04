from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import asyncio
import traceback
import pyaudio
import threading
import base64
import json
from google import genai
from google.genai import types

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

# Replace with your actual API key
GEMINI_API_KEY = "AIzaSyAOCk8-5OSa-J0T0o4PhRsc6qT7-ttCcc4"

# Language mappings
LANGUAGES = {
    'English': {'code': 'en-US'},
    'Spanish': {'code': 'es-ES'},
    'French': {'code': 'fr-FR'},
    'German': {'code': 'de-DE'},
    'Italian': {'code': 'it-IT'},
    'Portuguese': {'code': 'pt-BR'},
    'Hindi': {'code': 'hi-IN'},
    'Chinese': {'code': 'zh-CN'},
    'Japanese': {'code': 'ja-JP'},
    'Korean': {'code': 'ko-KR'},
    'Arabic': {'code': 'ar-SA'},
    'Russian': {'code': 'ru-RU'},
    'Tamil': {'code': 'ta-IN'}
}

# Voice mappings based on gender
VOICES = {
    'male': 'Fenrir',
    'female': 'Leda'
}

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=GEMINI_API_KEY,
)

pya = pyaudio.PyAudio()

class WebSpeechTranslator:
    def __init__(self, target_language, gender, socket_id):
        self.target_language = target_language
        self.gender = gender
        self.socket_id = socket_id
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.is_running = False
        self.tasks = []

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
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=f"You are a speech-to-text translator that automatically detects the source language and converts it into *colloquial, spoken {self.target_language}* (including mixing with the source language when natural).\n\nYour task involves three clear steps:\n1. **Transcribe** the user's speech *exactly* as spoken — do not paraphrase or interpret.\n2. **Detect** the source language automatically from the transcribed speech.\n3. **Translate** the transcribed text naturally into colloquial {self.target_language}, blending with the source language where appropriate.\n4. **Output only** the final translated sentence in {self.target_language}— nothing else.\n\nImportant Rules:\n- **Automatically detect** the source language - it could be English, Spanish, French, Hindi, or any other language.\n- Do **not** add any greetings, explanations, or commentary.\n- Do **not** generate any new content — only translate what the user actually said.\n- Do **not** change or modify the meaning in any way.\n- Do **not** fabricate or assume anything not explicitly spoken by the user.\n- Begin transcribing *only after* the user finishes speaking.\n- Your response must consist solely of the final translated sentence, ready to be spoken aloud.\n- Do **not** repeat the same sentences until the user speaks again.\n- Do **not** mention the detected source language in your response.\n\n⚠️ Do **not** include the original transcription, detected language, or any additional text — only return the translated sentence in {self.target_language}.")],
                role="user"
            ),
        )

    async def listen_audio(self):
        """Capture audio from microphone and send to the session"""
        try:
            mic_info = pya.get_default_input_device_info()
            self.audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            
            kwargs = {"exception_on_overflow": False} if __debug__ else {}
            
            while self.is_running:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
        except Exception as e:
            socketio.emit('error', {'message': f'Audio input error: {str(e)}'}, room=self.socket_id)

    async def send_audio(self):
        """Send audio data to the session"""
        try:
            while self.is_running:
                audio_data = await self.out_queue.get()
                await self.session.send(input=audio_data)
        except Exception as e:
            socketio.emit('error', {'message': f'Audio send error: {str(e)}'}, room=self.socket_id)

    async def receive_audio(self):
        """Receive audio responses from the session and queue for playback"""
        try:
            while self.is_running:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        socketio.emit('translation', {'text': text, 'language': self.target_language}, room=self.socket_id)

                # Clear audio queue on interruption
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
        except Exception as e:
            socketio.emit('error', {'message': f'Audio receive error: {str(e)}'}, room=self.socket_id)

    async def play_audio(self):
        """Play received audio responses"""
        try:
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            while self.is_running:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
        except Exception as e:
            socketio.emit('error', {'message': f'Audio playback error: {str(e)}'}, room=self.socket_id)

    async def run(self):
        """Main execution loop"""
        try:
            async with client.aio.live.connect(model=MODEL, config=self.get_config()) as session:
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self.is_running = True

                socketio.emit('status', {'message': f'Connected! Speak in any language to get {self.target_language} translation.'}, room=self.socket_id)

                # Create tasks for all audio processing
                tasks = await asyncio.gather(
                    self.send_audio(),
                    self.listen_audio(),
                    self.receive_audio(),
                    self.play_audio(),
                    return_exceptions=True
                )

        except Exception as e:
            socketio.emit('error', {'message': f'Session error: {str(e)}'}, room=self.socket_id)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.close()

# Global storage for active translators
active_translators = {}

@app.route('/')
def index():
    return render_template('index.html', languages=list(LANGUAGES.keys()))

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('status', {'message': 'Connected to server. Select a language to start.'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    if request.sid in active_translators:
        active_translators[request.sid].cleanup()
        del active_translators[request.sid]

@socketio.on('start_translation')
def handle_start_translation(data):
    target_language = data.get('language', 'English')
    gender = data.get('gender', 'female')
    
    if target_language not in LANGUAGES:
        emit('error', {'message': 'Invalid language selected'})
        return
    
    if gender not in VOICES:
        emit('error', {'message': 'Invalid gender selected'})
        return
    
    # Stop existing translator if any
    if request.sid in active_translators:
        active_translators[request.sid].cleanup()
    
    # Create new translator
    translator = WebSpeechTranslator(target_language, gender, request.sid)
    active_translators[request.sid] = translator
    
    # Run translator in background thread
    def run_translator():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(translator.run())
        except Exception as e:
            socketio.emit('error', {'message': f'Translator error: {str(e)}'}, room=request.sid)
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_translator)
    thread.daemon = True
    thread.start()

@socketio.on('stop_translation')
def handle_stop_translation():
    if request.sid in active_translators:
        active_translators[request.sid].cleanup()
        del active_translators[request.sid]
        emit('status', {'message': 'Translation stopped'})

if __name__ == '__main__':
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("Please replace 'YOUR_API_KEY_HERE' with your actual Gemini API key!")
        exit(1)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)