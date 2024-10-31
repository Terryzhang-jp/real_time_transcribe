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
from queue import Queue

# 初始化日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 在文件开头添加 GraphvizGenerator 的实例化
graph_generator = GraphvizGenerator()

class AudioQueueProcessor:
    def __init__(self):
        self.audio_queue = Queue()
        self.processing_status = {
            'current_index': 0,
            'queue_size': 0,
            'processing': False,
            'total_processed': 0
        }
        
        self.current_result = None
        self.has_new_result = False
        self.last_emit_time = time.time()
        self.emit_interval = 0.1
        
        self.last_transcription = ""
        self.current_session_text = ""   # 完整会话文本
        self.last_segment_text = ""      # 最新片段文本
        self.processed_segments = set()   # 已处理片段集合
        
        # 启动处理线程
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        logger.info("AudioQueueProcessor initialized")

    def _emit_results(self, force=False):
        """发送处理结果"""
        try:
            if self.current_result:
                result_data = {
                    'result': {
                        'transcription': self.current_result.get('transcription', ''),
                        'translation': self.current_result.get('translation', ''),
                        'corrected_text': self.current_result.get('corrected_text', '')
                    },
                    'queue_status': self.get_status()
                }
                
                logger.info(f"Emitting new result with status: {self.get_status()}")
                
                with app.app_context():
                    socketio.emit('transcription_result', result_data)
                
                if force:
                    self.current_result = None
                    
        except Exception as e:
            logger.error(f"Error in _emit_results: {str(e)}", exc_info=True)

    def add_to_queue(self, audio_item):
        """添加音频到处理队列"""
        try:
            # 验证音频项格式
            if not isinstance(audio_item, dict):
                raise ValueError("Audio item must be a dictionary")
                
            if 'audio_data' not in audio_item:
                raise ValueError("Audio item must contain 'audio_data'")
                
            if 'metadata' not in audio_item:
                raise ValueError("Audio item must contain 'metadata'")
            
            # 添加到队列
            self.audio_queue.put(audio_item)
            
            # 更新状态
            self.processing_status['queue_size'] = self.audio_queue.qsize()
            
            logger.info(f"Added audio to queue. Current size: {self.processing_status['queue_size']}")
            logger.debug(f"Audio item metadata: {audio_item['metadata']}")
            
        except Exception as e:
            logger.error(f"Error adding to queue: {str(e)}", exc_info=True)
            raise

    def _convert_audio(self, audio_data):
        """将音频数据转换为WAV格式"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
                webm_file.write(audio_data)
                webm_path = webm_file.name

            wav_path = webm_path.replace('.webm', '.wav')
            
            # 使用ffmpeg转换音频格式
            command = [
                'ffmpeg', '-i', webm_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y', wav_path
            ]
            
            subprocess.run(command, capture_output=True)
            
            # 删除临时的webm文件
            os.unlink(webm_path)
            
            return wav_path
        except Exception as e:
            logger.error(f"Audio conversion error: {str(e)}")
            raise

    def _transcribe_audio(self, audio_data, metadata):
        """转写音频并更新结果"""
        try:
            wav_path = self._convert_audio(audio_data)
            logger.debug(f"Audio converted to WAV: {wav_path}")
            
            segments, _ = model.transcribe(
                wav_path,
                language=metadata['language'],
                beam_size=5,
                initial_prompt=self.last_transcription
            )
            
            # 只收集新的转写内容
            new_transcription = ""
            for segment in segments:
                segment_id = f"{segment.start}_{segment.end}_{segment.text}"
                if segment_id not in self.processed_segments:
                    new_transcription += segment.text + " "
                    self.processed_segments.add(segment_id)
            
            if new_transcription.strip():
                try:
                    # 更新当前会话文本
                    self.current_session_text += new_transcription
                    self.last_segment_text = new_transcription.strip()
                    
                    # 只翻译新的部分
                    translation = translator.translate(self.last_segment_text)
                    
                    # 使用 GPT-4 修正完整的转写内容
                    with app.app_context():
                        corrected_text = chain.invoke({
                            "text": self.current_session_text.strip()
                        })
                    
                    # 生成并发送图表更新
                    if graph_generator.should_update():
                        try:
                            graph_image = graph_generator.process_text(self.current_session_text)
                            if graph_image:
                                with app.app_context():
                                    socketio.emit('graph_update', {
                                        'image_data': graph_image,
                                        'status': '图表已更新'
                                    })
                        except Exception as e:
                            logger.error(f"Error generating graph: {str(e)}")
                            with app.app_context():
                                socketio.emit('graph_update', {
                                    'error': str(e)
                                })
                    
                    self.current_result = {
                        'transcription': self.last_segment_text,
                        'translation': translation,
                        'corrected_text': corrected_text
                    }
                    
                    self._emit_results(force=True)
                    
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    
            # 清理临时文件
            if os.path.exists(wav_path):
                os.unlink(wav_path)
                
            return self.current_result or {
                'transcription': '',
                'translation': '',
                'corrected_text': ''
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            raise

    def _process_queue(self):
        """处理音频队列"""
        while True:
            try:
                if not self.audio_queue.empty():
                    # 更新处理状态
                    self.processing_status['processing'] = True
                    self.processing_status['queue_size'] = self.audio_queue.qsize()
                    
                    # 获取音频数据
                    audio_item = self.audio_queue.get()
                    
                    try:
                        # 处理音频
                        result = self._transcribe_audio(
                            audio_item['audio_data'],
                            audio_item['metadata']
                        )
                        
                        if result:
                            # 更新处理状态
                            self.processing_status['total_processed'] += 1
                            logger.info(f"Updated total processed: {self.processing_status['total_processed']}")
                            
                            # 发送结果
                            self.current_result = result
                            self._emit_results()
                            
                            # 更新当前索引
                            self.processing_status['current_index'] += 1
                            
                            # 单独发送状态更新
                            with app.app_context():
                                socketio.emit('status_update', self.get_status())
                                logger.info(f"Emitted status update: {self.get_status()}")
                    
                    except Exception as e:
                        logger.error(f"Error processing audio item: {str(e)}", exc_info=True)
                        
                else:
                    self.processing_status['processing'] = False
                    with app.app_context():
                        socketio.emit('status_update', self.get_status())
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}", exc_info=True)
                continue

    def get_status(self):
        """获取当前处理态"""
        return {
            'current_index': self.processing_status['current_index'],
            'queue_size': self.processing_status['queue_size'],
            'processing': self.processing_status['processing'],
            'total_processed': self.processing_status['total_processed']
        }

    def clear_session(self):
        """清除当前会话的所有内容"""
        self.processed_segments.clear()
        self.current_session_text = ""
        self.current_result = None
        self.last_transcription = ""

# 初始化 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# 初始化 SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 初始化模型和工具
model = WhisperModel("medium", device="cpu", compute_type="int8")
translator = GoogleTranslator(source='auto', target='zh-CN')
chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)
prompt = ChatPromptTemplate.from_template(
    """请析并修正以下完整对话内容的语法和专业词汇，保持对话的连贯性：

{text}

修正后的完整内容："""
)
chain = prompt | chat_model | StrOutputParser()

# 添加图表目录
CHART_DIR = os.path.join(app.root_path, 'static', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# 初始化音频处理器
audio_processor = AudioQueueProcessor()

# 路由定义
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        language = request.form.get('language', 'zh')
        
        logger.debug(f"Received audio data size: {len(audio_data)} bytes")
        logger.debug(f"Language setting: {language}")
        
        audio_item = {
            'audio_data': audio_data,
            'metadata': {
                'language': language,
                'timestamp': time.time()
            }
        }
        
        audio_processor.add_to_queue(audio_item)
        
        return jsonify({
            'status': 'success',
            'message': 'Audio added to processing queue',
            'queue_status': audio_processor.get_status()
        })
        
    except Exception as e:
        logger.error(f"Error processing transcribe request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected with ID: {request.sid}")
    emit('connection_status', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('result_received')
def handle_result_received(data):
    logger.info(f"Client acknowledged result receipt: {data}")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.root_path, filename)

@socketio.on('ping')
def handle_ping():
    emit('pong')

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)