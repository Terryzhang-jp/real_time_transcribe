from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import tempfile
import os
import subprocess
import logging
import base64
import time

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# 加载 Faster Whisper 模型
model = WhisperModel("medium", device="cpu", compute_type="int8")

# 初始化 Google Translator
translator = GoogleTranslator(source='auto', target='zh-CN')

# 初始化 OpenAI Chat 模型
chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# 创建 ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("请根据上下文修正以下文本的语法和专业词汇：\n\n{text}\n\n修正后的文本：")

# 创建 RunnableSequence
chain = prompt | chat_model | StrOutputParser()

# 保存最后一段音频的转写结果
last_transcription = ""
full_transcription = ""
last_correction_time = time.time()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global last_transcription, full_transcription, last_correction_time
    app.logger.info("Transcribe route called")
    if 'audio' not in request.files:
        app.logger.error("No audio file in request")
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    language = request.form.get('language', 'zh')  # 默认为中文
    
    temp_audio_path = ''
    wav_path = ''

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            temp_audio_path = temp_audio.name
            audio_file.save(temp_audio_path)

        app.logger.info(f"Audio file saved to {temp_audio_path}")
        app.logger.info(f"Audio file size: {os.path.getsize(temp_audio_path)} bytes")
        app.logger.info(f"Selected language: {language}")
        
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

        # 使用上下文进行转写，并传入选择的语言
        segments, _ = model.transcribe(wav_path, language=language, beam_size=5, initial_prompt=last_transcription)
        transcription = " ".join([segment.text for segment in segments])

        app.logger.info(f"Transcription result: {transcription}")

        # 更新上下文和完整转写
        last_transcription = transcription[-100:]  # 保留最后100个字符作为上下文
        full_transcription += transcription + " "

        # 翻译转写结果为中文
        if language != 'zh':
            translation = translator.translate(transcription)
            app.logger.info(f"Translation result: {translation}")
        else:
            translation = transcription

        # 检查是否需要进行GPT-4修正
        current_time = time.time()
        corrected_text = ""
        if current_time - last_correction_time >= 15:
            corrected_text = chain.invoke({"text": full_transcription})
            full_transcription = corrected_text  # 更新完整转写为修正后的文本
            last_correction_time = current_time

        return jsonify({
            'transcription': transcription, 
            'translation': translation,
            'corrected_text': corrected_text
        })
    except Exception as e:
        app.logger.error(f"Error during transcription or translation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

if __name__ == '__main__':
    app.run(debug=True)
