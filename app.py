from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os
import subprocess
import logging
import base64

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# 加载 Faster Whisper 模型
model = WhisperModel("medium", device="cpu", compute_type="int8")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    app.logger.info("Transcribe route called")
    if 'audio' not in request.files:
        app.logger.error("No audio file in request")
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    
    temp_audio_path = ''
    wav_path = ''

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            temp_audio_path = temp_audio.name
            audio_file.save(temp_audio_path)

        app.logger.info(f"Audio file saved to {temp_audio_path}")
        app.logger.info(f"Audio file size: {os.path.getsize(temp_audio_path)} bytes")
        
        # 添加这些行来查看文件的前100个字节
        with open(temp_audio_path, 'rb') as f:
            first_100_bytes = f.read(100)
            app.logger.info(f"First 100 bytes of the file: {base64.b64encode(first_100_bytes).decode()}")

        # 检查文件是否为空
        if os.path.getsize(temp_audio_path) == 0:
            app.logger.warning("Empty audio file received")
            return jsonify({'transcription': ''}), 200

        # 使用ffprobe来获取文件信息
        probe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=format_name,duration', '-of', 'default=noprint_wrappers=1:nokey=1', temp_audio_path]
        probe_result = subprocess.run(probe_command, capture_output=True, text=True)
        app.logger.info(f"FFprobe result: {probe_result.stdout}")

        # 使用ffmpeg将WebM转换为WAV
        wav_path = temp_audio_path.replace('.webm', '.wav')
        command = ['ffmpeg', '-y', '-i', temp_audio_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            app.logger.error(f"FFmpeg error: {result.stderr}")
            return jsonify({'error': 'Audio conversion failed'}), 500

        app.logger.info(f"Audio converted to WAV: {wav_path}")

        segments, _ = model.transcribe(wav_path, language="zh", beam_size=5)
        transcription = " ".join([segment.text for segment in segments])

        app.logger.info(f"Transcription result: {transcription}")

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
