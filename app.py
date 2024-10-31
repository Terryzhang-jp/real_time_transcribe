from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from models.graph_generator import GraphvizGenerator
import tempfile
import os
import subprocess
import logging
import base64
import time
import threading


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# 初始化 GraphvizGenerator
graph_generator = GraphvizGenerator()
logger.info("Initialized GraphvizGenerator")

# 添加一个用于存储图表的目录
CHART_DIR = os.path.join(app.root_path, 'static', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# 添加一个测试路由
@app.route('/test_graph')
def test_graph():
    try:
        test_text = "这是一个测试文本，用于验证图形生成功能。"
        base64_image = graph_generator.process_text(test_text)
        return jsonify({
            'success': bool(base64_image),
            'message': '图形生成成功' if base64_image else '图形生成失败'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('transcription_update')
def handle_transcription_update(data):
    logger.info("=== Starting Graph Generation Process ===")
    logger.info(f"Received transcription update with length: {len(data.get('transcription', ''))}")
    transcription = data.get('transcription', '')
    
    if not transcription:
        logger.warning("Empty transcription received")
        emit('graph_status', {'status': '没有接收到转写内容'})
        return
    
    logger.info(f"Processing transcription: {transcription[:100]}...")
    emit('graph_status', {'status': '正在处理转写内容...'})
    
    try:
        # 检查是否需要更新
        if not graph_generator.should_update():
            emit('graph_status', {
                'status': f'等待更新周期 (还需 {int(30 - (time.time() - graph_generator.last_update_time))} 秒)'
            })
            return
            
        emit('graph_status', {'status': '正在生成图表描述...'})
        # 生成图形
        base64_image = graph_generator.process_text(transcription)
        
        if base64_image:
            logger.info("Graph generated successfully")
            emit('graph_update', {
                'image_data': base64_image,
                'status': '图表生成成功'
            })
        else:
            logger.warning("No graph was generated")
            emit('graph_update', {
                'error': '无法生成图表，可能是内容不足或格式问题'
            })
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error generating graph: {error_msg}", exc_info=True)
        emit('graph_update', {
            'error': f'生成图表时出错: {error_msg}'
        })

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

        # 在转写完成后，发送 WebSocket 更新
        socketio.emit('transcript_update', {'text': full_transcription})

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

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.root_path, filename)

if __name__ == '__main__':
    socketio.run(app, debug=True)
