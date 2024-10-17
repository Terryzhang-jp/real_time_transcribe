from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os
import subprocess
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# 加载 Faster Whisper 模型
model = WhisperModel("medium", device="cpu", compute_type="int8")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    
    temp_audio_path = ''
    wav_path = ''

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            temp_audio_path = temp_audio.name
            audio_file.save(temp_audio_path)

        # 检查文件是否为空
        if os.path.getsize(temp_audio_path) == 0:
            return jsonify({'transcription': ''}), 200

        # 使用ffmpeg将WebM转换为WAV
        wav_path = temp_audio_path.replace('.webm', '.wav')
        command = ['ffmpeg', '-i', temp_audio_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            app.logger.error(f"FFmpeg error: {result.stderr}")
            return jsonify({'error': 'Audio conversion failed'}), 500

        segments, _ = model.transcribe(wav_path, language="zh", beam_size=5)
        transcription = " ".join([segment.text for segment in segments])

        return jsonify({'transcription': transcription})
    except Exception as e:
        app.logger.error(f"Error during transcription: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

if __name__ == '__main__':
    app.run(debug=True)
