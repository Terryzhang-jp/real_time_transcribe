// Constants
const VOLUME_THRESHOLD = -50;
const NOISE_THRESHOLD = -70;
const MIN_HUMAN_FREQ = 85;
const MAX_HUMAN_FREQ = 255;
const CONTINUITY_THRESHOLD = 500;
const RECORDING_INTERVAL = 5000;
const CHAR_THRESHOLD = 150;

// Global Variables
let audioContext;
let analyser;
let microphone;
let isListening = false;
let mediaRecorder;
let audioChunks = [];
let transcriptionBuffer = '';
let translationBuffer = '';
let lastVoiceDetectionTime = 0;
let continuityBuffer = new Array(300).fill(0);
let selectedLanguage = '';
let currentCharCount = 0;

// Socket Connection Setup
const socket = io({
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 1000,
    timeout: 60000,
    transports: ['websocket', 'polling']
});

// DOM Elements
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const clearButton = document.getElementById('clearButton');
const languageSelect = document.getElementById('languageSelect');
const gptLanguageSelect = document.getElementById('gptLanguageSelect');
const status = document.getElementById('status');
const frequencyDisplay = document.getElementById('frequency');
const continuityDisplay = document.getElementById('continuity');
const volumeDisplay = document.getElementById('volume');
const debugPanel = document.getElementById('debug-panel');
const debugInfo = document.getElementById('debug-info');

// Canvas Setup
const frequencyCanvas = document.getElementById('frequencyVisualizer');
const frequencyCtx = frequencyCanvas.getContext('2d');
const continuityCanvas = document.getElementById('continuityVisualizer');
const continuityCtx = continuityCanvas.getContext('2d');
const volumeCanvas = document.getElementById('volumeVisualizer');
const volumeCtx = volumeCanvas.getContext('2d');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeCanvasSizes();
    setupEventListeners();
});

window.addEventListener('resize', initializeCanvasSizes);

// Initialize Functions
function initializeCanvasSizes() {
    const canvases = [frequencyCanvas, continuityCanvas, volumeCanvas];
    canvases.forEach(canvas => {
        const container = canvas.parentElement;
        canvas.width = container.clientWidth;
        canvas.height = 150;
    });
}

function setupEventListeners() {
    // Language Selection
    languageSelect.onchange = () => {
        selectedLanguage = languageSelect.value;
        startButton.disabled = !selectedLanguage;
    };

    // Button Controls
    startButton.onclick = startRecording;
    stopButton.onclick = stopRecording;
    clearButton.onclick = clearResults;
    document.getElementById('toggleDebug').onclick = toggleDebug;
}

// Audio Processing Functions
async function startRecording() {
    if (!selectedLanguage) {
        alert('请先选择语言');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setupAudioContext(stream);
        setupMediaRecorder(stream);
        
        isListening = true;
        detectSound();

        startButton.disabled = true;
        stopButton.disabled = false;
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showError('无法访问麦克风，请检查权限设置。');
    }
}

function setupAudioContext(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    microphone = audioContext.createMediaStreamSource(stream);

    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;
    microphone.connect(analyser);
}

function setupMediaRecorder(stream) {
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        if (audioChunks.length > 0) {
            sendAudioToServer();
        }
        if (isListening) {
            startNewRecording();
        }
    };

    startNewRecording();
}

// Sound Detection and Visualization
function detectSound() {
    if (!isListening) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);
    analyser.getFloatFrequencyData(dataArray);

    const nyquist = audioContext.sampleRate / 2;
    const minIndex = Math.floor(MIN_HUMAN_FREQ * bufferLength / nyquist);
    const maxIndex = Math.ceil(MAX_HUMAN_FREQ * bufferLength / nyquist);

    let maxVolume = -Infinity;
    let maxFrequency = 0;
    for (let i = minIndex; i <= maxIndex; i++) {
        if (dataArray[i] > maxVolume) {
            maxVolume = dataArray[i];
            maxFrequency = i * nyquist / bufferLength;
        }
    }

    updateSoundVisualizations(dataArray, maxVolume, maxFrequency);
    requestAnimationFrame(detectSound);
}

// Socket Event Handlers
socket.on('connect', () => {
    console.log('Socket connected with ID:', socket.id);
    updateConnectionStatus('已连接');
    socket.emit('start_stream', {
        language: languageSelect.value,
        gpt_language: gptLanguageSelect.value
    });
});

socket.on('disconnect', (reason) => {
    console.log('Socket disconnected:', reason);
    updateConnectionStatus('已断开');
    setTimeout(() => {
        if (!socket.connected) {
            socket.connect();
        }
    }, 1000);
});

socket.on('transcription_result', handleTranscriptionResult);
socket.on('processing_status', handleProcessingStatus);
socket.on('processing_error', handleProcessingError);
socket.on('graph_update', handleGraphUpdate);
socket.on('connection_status', handleConnectionStatus);

// Utility Functions
function updateConnectionStatus(status) {
    const statusElement = document.getElementById('connection-status-text');
    statusElement.textContent = status;
    statusElement.style.color = status === '已连接' ? '#4CAF50' : '#dc3545';
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 3000);
}

function updateDebugInfo(info) {
    debugInfo.textContent = JSON.stringify(info, null, 2);
}

function toggleDebug() {
    debugPanel.style.display = debugPanel.style.display === 'none' ? 'block' : 'none';
}

// Result Handling Functions
function handleTranscriptionResult(data) {
    console.log('收到转写结果:', data);
    
    if (data.result) {
        updateTranscriptionText(data.result);
        updateCharCount(data.result);
        updateQueueStatus(data.queue_status);
    }
}

function updateTranscriptionText(result) {
    const transcriptionDiv = document.getElementById('transcription');
    const translationDiv = document.getElementById('translation');
    const correctedDiv = document.getElementById('corrected-text');
    
    if (result.transcription) {
        transcriptionDiv.textContent += (transcriptionDiv.textContent ? '\n' : '') + result.transcription;
        transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
    }
    
    if (result.translation) {
        translationDiv.textContent += (translationDiv.textContent ? '\n' : '') + result.translation;
        translationDiv.scrollTop = translationDiv.scrollHeight;
    }
    
    if (result.corrected_text) {
        correctedDiv.textContent = result.corrected_text;
        correctedDiv.scrollTop = correctedDiv.scrollHeight;
    }
}

// Cleanup Functions
function clearResults() {
    currentCharCount = 0;
    transcriptionBuffer = '';
    translationBuffer = '';
    document.getElementById('transcription').textContent = '';
    document.getElementById('translation').textContent = '';
    document.getElementById('corrected-text').textContent = '';
    document.getElementById('current-char-count').textContent = '0';
    document.getElementById('chars-until-update').textContent = CHAR_THRESHOLD;
    document.getElementById('char-progress').style.width = '0%';
    console.log('Cleared all results');
}

// Audio Processing Functions (continued)
function startNewRecording() {
    audioChunks = [];
    mediaRecorder.start();
    setTimeout(() => {
        if (mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    }, RECORDING_INTERVAL);
}

function sendAudioToServer() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    console.log('Sending audio blob:', audioBlob.size, 'bytes');
    
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('language', selectedLanguage);
    formData.append('gpt_language', document.getElementById('gptLanguageSelect').value);

    fetch('/transcribe', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Server response:', data);
        updateDebugInfo({ event: 'audio_upload', response: data });
        
        if (data.error) {
            showError(data.error);
        } else {
            updateQueueStatus(data.queue_status);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('服务器通信错误');
    });
}

// Visualization Functions
function updateSoundVisualizations(dataArray, maxVolume, maxFrequency) {
    visualizeFrequency(dataArray);
    visualizeVolume(maxVolume);
    updateVolumeDisplay(maxVolume);
    
    if (maxVolume > NOISE_THRESHOLD) {
        updateSoundStatus(maxVolume, maxFrequency);
    } else {
        updateSilenceStatus();
    }
    
    updateContinuityBuffer();
    visualizeContinuity();
}

function visualizeFrequency(dataArray) {
    frequencyCtx.fillStyle = 'rgb(200, 200, 200)';
    frequencyCtx.fillRect(0, 0, frequencyCanvas.width, frequencyCanvas.height);
    frequencyCtx.lineWidth = 2;
    frequencyCtx.strokeStyle = 'rgb(0, 0, 0)';
    frequencyCtx.beginPath();

    const sliceWidth = frequencyCanvas.width * 1.0 / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        const v = (dataArray[i] + 140) / 140;
        const y = v * frequencyCanvas.height / 2;

        if (i === 0) {
            frequencyCtx.moveTo(x, y);
        } else {
            frequencyCtx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    frequencyCtx.lineTo(frequencyCanvas.width, frequencyCanvas.height / 2);
    frequencyCtx.stroke();
}

function visualizeVolume(volume) {
    volumeCtx.clearRect(0, 0, volumeCanvas.width, volumeCanvas.height);
    volumeCtx.fillStyle = 'rgb(0, 255, 0)';
    const height = (volume + 140) * volumeCanvas.height / 140;
    volumeCtx.fillRect(0, volumeCanvas.height - height, volumeCanvas.width, height);
}

function visualizeContinuity() {
    continuityCtx.fillStyle = 'rgb(200, 200, 200)';
    continuityCtx.fillRect(0, 0, continuityCanvas.width, continuityCanvas.height);
    
    const barWidth = continuityCanvas.width / continuityBuffer.length;
    continuityCtx.fillStyle = 'rgb(0, 0, 0)';
    
    for (let i = 0; i < continuityBuffer.length; i++) {
        const barHeight = continuityBuffer[i] * continuityCanvas.height;
        continuityCtx.fillRect(
            i * barWidth,
            continuityCanvas.height - barHeight,
            barWidth,
            barHeight
        );
    }
}

// Status Update Functions
function updateSoundStatus(volumeDB, maxFrequency) {
    if (volumeDB > VOLUME_THRESHOLD) {
        status.textContent = '检测到人声';
        frequencyDisplay.textContent = `主要频率: ${maxFrequency.toFixed(2)} Hz`;
        
        const now = performance.now();
        const timeSinceLastVoice = now - lastVoiceDetectionTime;
        lastVoiceDetectionTime = now;

        continuityDisplay.textContent = timeSinceLastVoice < CONTINUITY_THRESHOLD ?
            '语音连续性: 连续' : '语音连续性: 断续';
    } else {
        updateSilenceStatus();
    }
}

function updateSilenceStatus() {
    status.textContent = '未检测到人声';
    frequencyDisplay.textContent = '主要频率: - Hz';
    continuityDisplay.textContent = '语音连续性: -';
}

function updateVolumeDisplay(volumeDB) {
    volumeDisplay.textContent = `当前音量: ${volumeDB.toFixed(2)} dB`;
}

function updateContinuityBuffer() {
    const now = performance.now();
    const timeSinceLastVoice = now - lastVoiceDetectionTime;
    continuityBuffer.push(timeSinceLastVoice < CONTINUITY_THRESHOLD ? 1 : 0);
    continuityBuffer.shift();
}

// Queue Status Functions
function updateQueueStatus(status) {
    if (status) {
        document.getElementById('queue-size').textContent = status.queue_size;
        document.getElementById('total-processed').textContent = status.total_processed;
        
        if (status.current_processing) {
            document.getElementById('current-processing').textContent = status.current_processing;
        }
        
        const progress = status.queue_size > 0 
            ? ((status.current_index / status.queue_size) * 100).toFixed(1)
            : 0;
        document.getElementById('process-progress').style.width = `${progress}%`;
    }
}

function updateCharCount(result) {
    if (result.transcription) {
        const newChars = result.transcription.length;
        currentCharCount += newChars;
        
        const charsUntilUpdate = CHAR_THRESHOLD - (currentCharCount % CHAR_THRESHOLD);
        const progress = ((currentCharCount % CHAR_THRESHOLD) / CHAR_THRESHOLD * 100).toFixed(1);
        
        document.getElementById('current-char-count').textContent = currentCharCount;
        document.getElementById('chars-until-update').textContent = charsUntilUpdate;
        document.getElementById('char-progress').style.width = `${progress}%`;
    }
}

// Graph Update Handler
function handleGraphUpdate(data) {
    console.log('Received graph update:', data);
    const graphImg = document.getElementById('logic-graph');
    const statusDiv = document.getElementById('graph-status');
    const currentStatus = document.getElementById('current-status');
    const errorMessage = document.getElementById('error-message-graph');
    const lastUpdateTime = document.getElementById('last-update-time');
    
    if (data.image_data) {
        graphImg.src = data.image_data;
        graphImg.style.display = 'block';
        statusDiv.textContent = '图表已更新';
        currentStatus.textContent = '更新成功';
        errorMessage.textContent = '';
        lastUpdateTime.textContent = new Date().toLocaleTimeString();
        
        // Reset char count for new cycle
        currentCharCount = currentCharCount % CHAR_THRESHOLD;
        updateCharCount({ transcription: '' });
    } else if (data.status) {
        statusDiv.textContent = data.status;
        currentStatus.textContent = data.status;
    } else if (data.error) {
        statusDiv.textContent = '图表生成失败';
        currentStatus.textContent = '生成失败';
        errorMessage.textContent = data.error;
        console.error('Graph generation error:', data.error);
    }
}

// Connection Status Handler
function handleConnectionStatus(data) {
    console.log('Connected to server:', data);
    const deviceStatus = document.querySelector('#deviceInfo .device-status');
    const modelStatus = document.querySelector('#modelInfo .model-status');
    
    if (data.device_info) {
        deviceStatus.textContent = data.device_info.device_info;
        deviceStatus.className = 'device-status ' + 
            (data.device_info.device === 'cuda' ? 'gpu' : 'cpu');
        modelStatus.textContent = `计算类型: ${data.device_info.compute_type}`;
    }
}

// Processing Status Handler
function handleProcessingStatus(data) {
    console.log('处理状态更新:', data);
    updateQueueStatus(data);
}

// Error Handler
function handleProcessingError(data) {
    console.error('处理错误:', data);
    showError(`处理错误: ${data.error}`);
}

// Stop Recording Function
function stopRecording() {
    socket.emit('stop_recording');
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
    if (microphone) {
        microphone.disconnect();
    }
    if (audioContext) {
        audioContext.close();
    }
    isListening = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    status.textContent = '等待队列处理完成...';
}

// Heart Beat Check
setInterval(() => {
    if (socket.connected) {
        socket.emit('ping');
    }
}, 30000);

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeCanvasSizes();
    setupEventListeners();
});