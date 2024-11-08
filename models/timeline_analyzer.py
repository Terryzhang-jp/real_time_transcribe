# This is timeline_analyzer.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import json
import logging
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TimelineAnalyzer:
    """
    会议时间轴分析器
    负责分析会议内容并生成时间轴式的会议记录
    """
    
    def __init__(self):
        """初始化时间轴分析器"""
        # 初始化 GPT 模型
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)
        
        # 存储会议记录
        self.timeline_segments: List[Dict[str, Any]] = []
        
        # 记录当前状态
        self.current_phase: Optional[str] = None
        self.char_count: int = 0
        self.CHAR_THRESHOLD: int = 150
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_template(
            """你是一个专业的会议记录分析专家。现在你需要分析一段会议内容。

            背景信息：
            - 这是一个业务会议的实时记录
            - 当前时间段：{start_time} - {end_time}
            - 之前的会议阶段：{previous_phase}

            当前文本内容：
            {text}

            请以JSON格式返回分析结果，要求：
            1. phase: 当前会议所处阶段（自由判断，如：开场寒暄/需求讨论/技术探讨/总结等）
            2. main_point: 当前时间段的主要内容
            3. details: 包含具体的要点和展开内容的数组
            4. continuation: 与上一阶段的关系（是否继续上一个话题，或开启新话题）

            仅返回JSON格式的结果， 并且输出使用英语。
            """
        )

    def analyze_segment(self, text: str, start_time: str, end_time: str) -> dict:
        try:
            response = self._get_gpt_analysis(text, start_time, end_time)
            logger.debug(f"Raw GPT response: {response}")  # 添加日志
            
            # 确保响应是 JSON 格式
            if not response.strip().startswith('{'):
                logger.error(f"Invalid GPT response format: {response}")
                return {}
                
            analysis = json.loads(response)
            logger.info(f"Parsed analysis: {analysis}")
            
            # 更新时间轴数据
            self.timeline_segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'phase': analysis.get('phase', '未知阶段'),
                'main_point': analysis.get('main_point', ''),
                'details': analysis.get('details', []),
                'raw_text': text
            })
            
            return analysis
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return {}
        
    def _get_gpt_analysis(self, text: str, start_time: str, end_time: str) -> str:
        prompt = f"""你是一个专业的会议记录分析专家。分析以下会议内容并以JSON格式返回结果。
        
        背景信息：
        - 这是一个业务会议的实时记录
        - 当前时间段：{start_time} - {end_time}
        
        当前文本内容：
        {text}
        
        请严格按照以下JSON格式返回分析结果：
        {{
            "phase": "当前会议阶段",
            "main_point": "主要内容",
            "details": ["要点1", "要点2"],
            "continuation": "与上一阶段的关系"
        }}
        
        只返回JSON格式的结果，不要添加其他说明文字, 并且输出使用英语。
        """
        
        try:
            response = self.llm.invoke(prompt).content
            logger.debug(f"GPT response: {response}")
            return response
        except Exception as e:
            logger.error(f"GPT analysis error: {str(e)}", exc_info=True)
            return "{}"
        
    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        验证分析结果格式
        
        Args:
            analysis: 需要验证的分析结果
            
        Returns:
            bool: 验证是否通过
        """
        try:
            required_fields = ['phase', 'main_point', 'details', 'continuation']
            
            # 检查必需字段
            if not all(field in analysis for field in required_fields):
                return False
                
            # 检查字段类型
            if not isinstance(analysis['phase'], str):
                return False
            if not isinstance(analysis['main_point'], str):
                return False
            if not isinstance(analysis['details'], list):
                return False
            if not isinstance(analysis['continuation'], bool):
                return False
                
            # 检查内容不为空
            if not analysis['phase'].strip():
                return False
            if not analysis['main_point'].strip():
                return False
            if not analysis['details']:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False
            
    def _get_default_analysis(self) -> Dict[str, Any]:
        """
        返回默认的分析结果
        
        Returns:
            Dict: 默认的分析结果
        """
        return {
            "phase": "会议进行中",
            "main_point": "内容分析中",
            "details": ["等待更多内容..."],
            "continuation": False
        }
        
    def _update_timeline(self, analysis: Dict[str, Any], start_time: str, 
                        end_time: str, raw_text: str) -> None:
        """
        更新时间轴数据
        
        Args:
            analysis: 分析结果
            start_time: 开始时间
            end_time: 结束时间
            raw_text: 原始文本
        """
        segment = {
            "timestamp": {
                "start": start_time,
                "end": end_time
            },
            "phase": analysis["phase"],
            "content": {
                "main_point": analysis["main_point"],
                "details": analysis["details"]
            },
            "raw_text": raw_text,
            "continuation": analysis["continuation"]
        }
        
        self.timeline_segments.append(segment)
        self.current_phase = analysis["phase"]
        
        logger.debug(f"Timeline updated with new segment: {segment['timestamp']}")
        
    def get_timeline(self) -> Dict[str, Any]:
        """
        获取完整的时间轴数据
        
        Returns:
            Dict: 完整的时间轴数据
        """
        return {
            "segments": self.timeline_segments,
            "current_phase": self.current_phase
        }
        
    def should_analyze(self, text_length: int) -> bool:
        """
        检查是否需要进行新的分析
        
        Args:
            text_length: 新增文本的长度
            
        Returns:
            bool: 是否需要进行分析
        """
        self.char_count += text_length
        logger.debug(f"Current char count: {self.char_count}, threshold: {self.CHAR_THRESHOLD}")
        if self.char_count >= self.CHAR_THRESHOLD:
            self.char_count = 0
            logger.info("Analysis threshold reached")
            return True
        return False
        
    def clear_timeline(self) -> None:
        """清除时间轴数据"""
        self.timeline_segments = []
        self.current_phase = None
        self.char_count = 0
        logger.info("Timeline cleared")
        
    def get_remaining_chars(self) -> int:
        """
        获取距离下次分析还需要的字符数
        
        Returns:
            int: 剩余需要的字符数
        """
        return self.CHAR_THRESHOLD - self.char_count
        
    def get_analysis_stats(self) -> Dict[str, Any]:
        """
        获取分析统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "total_segments": len(self.timeline_segments),
            "current_phase": self.current_phase,
            "char_count": self.char_count,
            "chars_until_next": self.get_remaining_chars()
        }
