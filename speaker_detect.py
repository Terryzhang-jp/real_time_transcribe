from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import logging
import librosa
from collections import deque
import time
from pyannote.audio import Pipeline
import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
recent_audio = deque(maxlen=10*SAMPLE_RATE)  # 存储10秒的音频
current_speaker = None
last_speaker_time = 0
SPEAKER_TIMEOUT = 3  # 3秒内保持同一说话人

# 初始化 pyannote.audio pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="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")

@app.route('/')
def index():
    return render_template('speaker_detect.html')

def process_audio(audio_signal, original_sample_rate):
    global current_speaker, last_speaker_time

    logger.info(f"Processing audio of length: {len(audio_signal)}")
    start_time = time.time()

    # 如果需要，将音频重采样到 16000 Hz
    if original_sample_rate != SAMPLE_RATE:
        logger.info(f"Resampling audio from {original_sample_rate} Hz to {SAMPLE_RATE} Hz")
        audio_signal = librosa.resample(audio_signal, orig_sr=original_sample_rate, target_sr=SAMPLE_RATE)
        logger.info(f"Resampled audio length: {len(audio_signal)}")

    # 将音频转换为 pyannote.audio 所需的格式
    waveform = torch.from_numpy(audio_signal).float().unsqueeze(0)

    logger.info("Starting pyannote.audio diarization")
    # 使用 pyannote.audio 进行说话人分割和识别
    diarization = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})

    # 获取最后一个说话人
    last_speaker = None
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        last_speaker = speaker
        logger.info(f"Detected speech from {speaker} at {turn}")

    current_time = time.time()
    if last_speaker:
        if last_speaker != current_speaker or current_time - last_speaker_time > SPEAKER_TIMEOUT:
            current_speaker = last_speaker
            last_speaker_time = current_time
        logger.info(f"Current speaker updated to: {current_speaker}")
    else:
        current_speaker = None
        logger.info("No speaker detected in this segment")

    processing_time = time.time() - start_time
    logger.info(f"Audio processing completed in {processing_time:.2f} seconds")

    return current_speaker, list(diarization.labels())

@socketio.on('audio_data')
def handle_audio_data(data):
    global recent_audio

    try:
        audio_data = np.array(data['audio'], dtype=np.float32)
        sample_rate = data.get('sample_rate', SAMPLE_RATE)
        logger.info(f"Received audio data of shape: {audio_data.shape}, sample rate: {sample_rate}")

        # 将新的音频数据添加到 recent_audio
        recent_audio.extend(audio_data)
        logger.info(f"Total accumulated audio length: {len(recent_audio)}")

        # 如果累积了足够的音频数据，进行处理
        if len(recent_audio) >= 5 * sample_rate:
            logger.info("Processing accumulated audio")
            audio_to_process = np.array(recent_audio)
            current_speaker, all_speakers = process_audio(audio_to_process, sample_rate)

            logger.info(f"Speech detected. Current speaker: {current_speaker}")
            logger.debug(f"All speakers: {all_speakers}")
        
            emit('update_speakers', {
                'speakers': all_speakers,
                'current_speaker': current_speaker,
                'debug_info': {
                    'energy': float(np.mean(np.abs(audio_to_process))),
                    'max_amplitude': float(np.max(audio_to_process)),
                    'audio_length': len(audio_to_process)
                }
            })

            # 清空 recent_audio，只保留最后2秒的数据
            recent_audio = deque(list(recent_audio)[-2*int(sample_rate):], maxlen=10*SAMPLE_RATE)
            logger.info(f"Cleared audio buffer. Remaining audio length: {len(recent_audio)}")
        else:
            logger.info("Not enough audio data accumulated yet")

    except Exception as e:
        logger.error(f"Error processing audio data: {str(e)}")

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
