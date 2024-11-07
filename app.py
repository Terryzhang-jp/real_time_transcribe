# 导入必要的库和模块
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from faster_whisper import WhisperModel  # 用于语音识别
from deep_translator import GoogleTranslator  # 用于文本翻译
from langchain_openai import ChatOpenAI  # OpenAI的ChatGPT接口
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from models.graph_generator import GraphvizGenerator  # 图表生成器
import tempfile  # 临时文件处理
import os
import subprocess  # 用于执行系统命令
import logging  # 日志处理
import base64
import time
import threading
from queue import Queue  # 线程安全的队列
import torch
from typing import Dict, Any
from models.timeline_analyzer import TimelineAnalyzer

# 初始化日志配置
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 初始化图表生成器
graph_generator = GraphvizGenerator()

# 添加GPU检测函数
def get_device_info() -> Dict[str, Any]:
    """
    检测可用的计算设备并返回相关信息
    Returns:
        Dict包含设备类型和详细信息
    """
    if torch.cuda.is_available():
        return {
            'device': 'cuda',
            'device_name': torch.cuda.get_device_name(0),
            'compute_type': 'float16',
            'device_info': f"GPU: {torch.cuda.get_device_name(0)}",
            'device_index': 0
        }
    else:
        return {
            'device': 'cpu',
            'device_name': 'CPU',
            'compute_type': 'int8',
            'device_info': "CPU Mode",
            'device_index': 0  # CPU模式下也设置为0
        }

class AudioQueueProcessor:
    """
    音频队列处理器类
    负责管理和处理音频转写请求队列
    """
    def __init__(self):
        """
        初始化音频处理器
        设置队列和态追踪变量
        """
        # 初始化音频处理队列
        self.audio_queue = Queue()
        # 处理状态追踪
        self.processing_status = {
            'current_index': 0,    # 当前处理索引
            'queue_size': 0,       # 队列大小
            'processing': False,    # 是否正在处理
            'total_processed': 0,    # 已处理总数
            'is_recording': True  # 新增录音状态标志
        }

        # 结果相变量
        self.current_result = None  # 当前处理结果
        self.has_new_result = False  # 是否有新结果
        self.last_emit_time = time.time()  # 上次发送结果时间
        self.emit_interval = 0.1  # 发送间隔时间

        # 转写相关变量
        self.last_transcription = ""  # 上次的转写结果
        self.current_session_text = ""  # 当前会话的完整文本
        self.last_segment_text = ""  # 最新的文本片段
        self.processed_segments = set()  # 已处理的片段集合

        # 启动处理线程
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        logger.info("AudioQueueProcessor initialized")

        self.text_buffer = ""  # 存储累积的文本
        self.char_count = 0    # 字符计数器
        self.CHAR_THRESHOLD = 150  # 设置字符阈值为150

        # 现有的初始化代码...
        self.device_info = get_device_info()
        logger.info(f"Using device: {self.device_info['device_info']}")

        # 添加语言提示模板
        self.language_prompts = {
            'zh': '中文',
            'en': 'English',
            'ja': '日本語で'
        }
        
        # 初始化 chat model
        self.chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)

        # 添加时间轴分析器
        self.timeline_analyzer = TimelineAnalyzer()
        self.current_segment_text = ""
        self.current_segment_start_time = None
        logger.info("TimelineAnalyzer initialized")
        self.current_segment_text = ""

    def _emit_results(self, force=False):
        """
        发送处理结果到客户端
        Args:
            force: 是否强制发送结果
        """
        try:
            if self.current_result:
                # 准备发送的数据
                result_data = {
                    'result': {
                        'transcription': self.current_result.get('transcription', ''),
                        'translation': self.current_result.get('translation', ''),
                        'corrected_text': self.current_result.get('corrected_text', '')
                    },
                    'queue_status': self.get_status()
                }

                logger.info(f"Emitting new result with status: {self.get_status()}")

                # 在Flask上下文中发送结果
                with app.app_context():
                    socketio.emit('transcription_result', result_data)

                if force:
                    self.current_result = None

        except Exception as e:
            logger.error(f"Error in _emit_results: {str(e)}", exc_info=True)

    def add_to_queue(self, audio_item):
        """
        添加音频到处理队列
        Args:
            audio_item: 包含音频数据的字典
        """
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
            logger.error(f"Error adding to queue: {str(e)}")
            raise

    def _convert_audio(self, audio_data):
        """
        将WebM音频数据转换为WAV格式
        Args:
            audio_data: 原始音频数据
        Returns:
            转换后的WAV文件路径
        """
        try:
            # 创建临时WebM文件
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
        """
        转写音频并处理结果
        Args:
            audio_data: 音频数据
            metadata: 元数据信息
        Returns:
            处理结果字典
        """
        try:
            # 转换音频格式
            wav_path = self._convert_audio(audio_data)
            logger.debug(f"Audio converted to WAV: {wav_path}")
            logger.info(f"Transcribing using {self.device_info['device_info']}")

            # 使用Whisper模型进转写
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
                    
                    # 记录段落开始时间
                    if self.current_segment_start_time is None:
                        self.current_segment_start_time = metadata['timestamp']
                        logger.debug(f"Starting new segment at {self.current_segment_start_time}")
                    
                    # 累积当前段落文本
                    self.current_segment_text += new_transcription
                    logger.debug(f"Current segment length: {len(self.current_segment_text)}")
                    
                    # 检查是否需要进行时间轴分析
                    if hasattr(self, 'timeline_analyzer') and self.timeline_analyzer.should_analyze(len(new_transcription)):
                        logger.info("Triggering timeline analysis")
                        try:
                            analysis = self.timeline_analyzer.analyze_segment(
                                text=self.current_segment_text,
                                start_time=self._format_timestamp(self.current_segment_start_time),
                                end_time=self._format_timestamp(time.time())
                            )
                            
                            # 发送时间轴更新
                            self._emit_timeline_update()
                            
                            # 重置段落
                            self.current_segment_text = ""
                            self.current_segment_start_time = None
                        except Exception as e:
                            logger.error(f"Timeline analysis error: {str(e)}", exc_info=True)
                    
                    # 翻译新的部分
                    translation = translator.translate(self.last_segment_text)

                    # 获取语言设置
                    gpt_language = metadata.get('gpt_language', 'zh')
                    language_prompt = self.language_prompts.get(gpt_language, self.language_prompts['zh'])
                    
                    prompt = ChatPromptTemplate.from_template(
                    """请分析并修正以下完整对话内容的语法和专业词汇，保持对话的连贯性：

                    {text}

                    请使用{language_prompt}输出 修正后的完整内容 只输出修正后的内容：
                    
                    """
                    )
                    
                    # 创建处理链
                    chain = prompt | self.chat_model | StrOutputParser()
                    
                    # 调用 GPT
                    corrected_text = chain.invoke({
                        "text": self.current_session_text.strip(),
                        "language_prompt": language_prompt
                    })

                    # 生成并发送图表更新
                    if graph_generator.should_generate_graph(len(new_transcription)):
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

                    # 设置当前结果
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
        """
        处理音频队列的主循环
        持续监控队列并处理新的音频项
        """
        while True:
            try:
                if not self.audio_queue.empty():
                    self.processing_status['processing'] = True
                    self.processing_status['queue_size'] = self.audio_queue.qsize()

                    audio_item = self.audio_queue.get()

                    try:
                        result = self._transcribe_audio(
                            audio_item['audio_data'],
                            audio_item['metadata']
                        )

                        if result:
                            self.processing_status['total_processed'] += 1
                            logger.info(f"Updated total processed: {self.processing_status['total_processed']}")

                            self.current_result = result
                            self._emit_results()

                            self.processing_status['current_index'] += 1

                            with app.app_context():
                                socketio.emit('status_update', self.get_status())
                                
                                # 如果队列为空且不在录音，发送完成信号
                                if self.audio_queue.empty() and not self.processing_status['is_recording']:
                                    socketio.emit('processing_complete', {
                                        'message': '所有音频处理完成',
                                        'final_status': self.get_status()
                                    })

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
        """
        获取当前处理状态
        Returns:
            包含处理状态信息的字典
        """
        return {
            'current_index': self.processing_status['current_index'],
            'queue_size': self.processing_status['queue_size'],
            'processing': self.processing_status['processing'],
            'total_processed': self.processing_status['total_processed']
        }

    def clear_session(self):
        """
        清当前会话的所有内容
        重置所有会话相关的变量
        """
        self.processed_segments.clear()
        self.current_session_text = ""
        self.current_result = None
        self.last_transcription = ""
        self.timeline_analyzer.clear_timeline()
        self.current_segment_start_time = None
        self.current_segment_text = ""

    def process_transcription_result(self, text):
        """处理转写结果"""
        # 累积文本
        self.text_buffer += text
        
        # 当累积字符数达到阈值时触发图表生成
        if graph_generator.should_generate_graph(len(text)):
            try:
                # 生成图表
                graph_image = graph_generator.process_text(self.text_buffer)
                
                if graph_image:
                    socketio.emit('graph_update', {
                        'image_data': graph_image,
                        'status': '图表已更新',
                        'text_length': len(self.text_buffer)
                    })
            except Exception as e:
                logger.error(f"Error generating graph: {str(e)}")
                socketio.emit('graph_update', {
                    'error': str(e)
                })

    def _process_audio(self, audio_item):
        """处理单个音频项"""
        try:
            # 获取转写结果
            result = self._transcribe_audio(audio_item['audio_data'], audio_item['metadata'])
            return result
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            return None
            
    def _emit_timeline_update(self):
        """发送时间轴更新事件"""
        try:
            timeline_data = self.timeline_analyzer.get_timeline()
            stats = self.timeline_analyzer.get_analysis_stats()
            
            logger.info(f"Emitting timeline update with {len(timeline_data.get('segments', []))} segments")
            
            with app.app_context():
                socketio.emit('timeline_update', {
                    'timeline': timeline_data,
                    'stats': stats
                })
        except Exception as e:
            logger.error(f"Error in _emit_timeline_update: {str(e)}", exc_info=True)
            
    def _format_timestamp(self, timestamp):
        """格式化时间戳"""
        return time.strftime('%H:%M:%S', time.localtime(timestamp))

# 初始化Flask应用和配置
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# 初始化WebSocket支持
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 初始化必要的模型和工具
device_info = get_device_info()

# 简化模初始化
if device_info['device'] == 'cuda':
    model = WhisperModel("medium", device="cuda", compute_type="float16")
else:
    model = WhisperModel("medium", device="cpu", compute_type="int8")
translator = GoogleTranslator(source='auto', target='zh-CN')  # 翻译器
chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)  # GPT模型


# 设置图表存储目录
CHART_DIR = os.path.join(app.root_path, 'static', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# 初始化音频处理器
audio_processor = AudioQueueProcessor()

# 路由定义
@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html', device_info=device_info)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    处理音频转写请求的路由
    接收音频文件并添加到处队列
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        audio_data = audio_file.read()
        language = request.form.get('language', 'zh')
        gpt_language = request.form.get('gpt_language', 'zh')

        logger.debug(f"Received audio data size: {len(audio_data)} bytes")
        logger.debug(f"Language setting: {language}")
        logger.debug(f"GPT language setting: {gpt_language}")

        # 准备音频数据
        audio_item = {
            'audio_data': audio_data,
            'metadata': {
                'language': language,
                'gpt_language': gpt_language,
                'timestamp': time.time()
            }
        }

        # 添加到处理队列
        audio_processor.add_to_queue(audio_item)

        return jsonify({
            'status': 'success',
            'message': 'Audio added to processing queue',
            'queue_status': audio_processor.get_status()
        })

    except Exception as e:
        logger.error(f"Error processing transcribe request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """处理客户端连接事件"""
    logger.info(f"Client connected with ID: {request.sid}")
    device_info = get_device_info()  # 获取最新的设备信息
    emit('connection_status', {
        'status': 'connected', 
        'sid': request.sid,
        'device_info': {
            'device': device_info['device'],
            'device_info': device_info['device_info'],
            'compute_type': device_info['compute_type'],
            'device_name': device_info['device_name']
        }
    })
    
    # 发送当前时间轴状态（如果有）
    if hasattr(audio_processor, 'timeline_analyzer'):
        timeline_data = audio_processor.timeline_analyzer.get_timeline()
        stats = audio_processor.timeline_analyzer.get_analysis_stats()
        emit('timeline_update', {
            'timeline': timeline_data,
            'stats': stats,
            'status': audio_processor.get_status()
        })

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接事件"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('result_received')
def handle_result_received(data):
    """处理客���端结果接收确认"""
    logger.info(f"Client acknowledged result receipt: {data}")

@socketio.on('timeline_request')
def handle_timeline_request():
    """处理时间轴数据请求"""
    try:
        timeline_data = audio_processor.timeline_analyzer.get_timeline()
        stats = audio_processor.timeline_analyzer.get_analysis_stats()
        
        emit('timeline_update', {
            'timeline': timeline_data,
            'stats': stats,
            'status': audio_processor.get_status()
        })
    except Exception as e:
        logger.error(f"Error handling timeline request: {str(e)}", exc_info=True)
        emit('processing_error', {'error': str(e)})

# 静态文件服务
@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory(app.root_path, filename)

# 心跳检测
@socketio.on('ping')
def handle_ping():
    """处理客户端ping请求"""
    emit('pong')

@socketio.on('stop_recording')
def handle_stop_recording():
    audio_processor.processing_status['is_recording'] = False
    logger.info("Recording stopped, continuing to process remaining queue")

@app.route('/clear', methods=['POST'])
def clear_session():
    """清除当前会话数据"""
    try:
        audio_processor.clear_session()
        return jsonify({'status': 'success', 'message': '会话已清除'})
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# 主程序入口
if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)