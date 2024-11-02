from graphviz import Digraph
import base64
import io
from langchain_openai import ChatOpenAI
import time
import logging
import json
import copy

logger = logging.getLogger(__name__)

class MeetingAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini")
        
    def analyze_meeting(self, text):
        prompt = f"""
        角色：你是一个专业的会议内容分析专家。请分析以下文本，提取主题和关键论点。

        文本内容：
        {text}

        请以JSON格式返回以下内容：
        1. 主题（main_topic）：提取或总结一个核心主题
        2. 关键论点（key_points）：提取2-3个主要论点，每个论点可以有1-2个支持性的细节或例子

        输出格式：
        {{
            "main_topic": "主题内容",
            "key_points": [
                {{
                    "point": "关键论点1",
                    "supporting_details": ["支持性细节1", "支持性细节2"]
                }},
                {{
                    "point": "关键论点2",
                    "supporting_details": ["支持性细节1"]
                }}
            ]
        }}

        要求：
        1. 保持JSON格式的合法性
        2. 主题简明扼要
        3. 关键论点要抓住重点
        4. 支持性细节要具体
        5. 直接输出JSON，不要其他说明文字
        """
        
        try:
            response = self.llm.predict(prompt)
            analysis = json.loads(response)
            return analysis if self._validate_analysis(analysis) else self._get_default_analysis()
        except Exception as e:
            logger.error(f"Meeting analysis error: {str(e)}")
            return self._get_default_analysis()

    def _validate_analysis(self, analysis):
        """验证分析结果格式"""
        try:
            if not isinstance(analysis, dict):
                return False
            
            if 'main_topic' not in analysis or not analysis['main_topic']:
                return False
                
            if 'key_points' not in analysis or not isinstance(analysis['key_points'], list):
                return False
                
            for point in analysis['key_points']:
                if 'point' not in point or 'supporting_details' not in point:
                    return False
                if not isinstance(point['supporting_details'], list):
                    return False
                    
            return True
        except Exception:
            return False

    def _get_default_analysis(self):
        """返回默认的分析结果"""
        return {
            "main_topic": "会议进行中",
            "key_points": [
                {
                    "point": "正在收集要点",
                    "supporting_details": ["等待更多内容..."]
                }
            ]
        }

class GraphvizGenerator:
    def __init__(self):
        self.analyzer = MeetingAnalyzer()
        self.last_analysis = None
        self.current_char_count = 0
        self.CHAR_THRESHOLD = 150

    def generate_graph_instruction(self, analysis):
        """根据分析结果生成图表指令"""
        try:
            # 生成唯一的节点ID
            main_id = 'main'
            nodes = [{
                'id': main_id,
                'label': self._format_label(analysis['main_topic']),
                'style': {
                    'shape': 'box',
                    'style': 'filled,rounded',
                    'fillcolor': 'lightblue',
                    'fontsize': '14',
                    'width': '2',
                    'height': '0.7'
                }
            }]
            
            edges = []
            
            # 为每个关键点创建节点和边
            for idx, point in enumerate(analysis['key_points']):
                point_id = f'point_{idx}'
                nodes.append({
                    'id': point_id,
                    'label': self._format_label(point['point']),
                    'style': {
                        'shape': 'box',
                        'style': 'filled,rounded',
                        'fillcolor': 'lightgreen',
                        'fontsize': '12',
                        'width': '2'
                    }
                })
                
                # 连接主题到关键点
                edges.append({
                    'from': main_id,
                    'to': point_id,
                    'style': {
                        'color': 'black',
                        'arrowhead': 'normal',
                        'penwidth': '1.5'
                    }
                })
                
                # 为支持细节创建节点和边
                for detail_idx, detail in enumerate(point['supporting_details']):
                    detail_id = f'detail_{idx}_{detail_idx}'
                    nodes.append({
                        'id': detail_id,
                        'label': self._format_label(detail),
                        'style': {
                            'shape': 'box',
                            'style': 'filled,rounded',
                            'fillcolor': 'white',
                            'fontsize': '10',
                            'width': '1.8'
                        }
                    })
                    
                    # 连接关键点到支持细节
                    edges.append({
                        'from': point_id,
                        'to': detail_id,
                        'style': {
                            'color': 'gray',
                            'arrowhead': 'normal',
                            'style': 'dashed',
                            'penwidth': '1.0'
                        }
                    })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'graph_attrs': {
                    'rankdir': 'TB',
                    'splines': 'ortho',
                    'nodesep': '0.5',
                    'ranksep': '0.7'
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating graph instruction: {str(e)}")
            return None

    def create_graph_from_instruction(self, instruction):
        """根据指令创建图表"""
        try:
            dot = Digraph(comment='Meeting Analysis')
            
            # 设置图表属性
            for key, value in instruction['graph_attrs'].items():
                dot.attr(key=key, value=value)
            
            # 创建所有节点
            for node in instruction['nodes']:
                style = node.get('style', {})
                dot.node(node['id'], node['label'], **style)
            
            # 创建所有边
            for edge in instruction['edges']:
                style = edge.get('style', {})
                dot.edge(edge['from'], edge['to'], '', **style)
            
            return dot
            
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            return self._create_default_graph()

    def _format_label(self, text, max_length=20):
        """格式化标签文本，添加换行"""
        if len(text) <= max_length:
            return text
            
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\\n'.join(lines)

    def process_text(self, current_text):
        """处理累积的文本并生成图表"""
        if not current_text.strip():
            return self._create_default_graph_image()
            
        try:
            # 使用完整文本进行分析
            analysis = self.analyzer.analyze_meeting(current_text)
            if not analysis:
                logger.warning("Analysis failed")
                if self.last_analysis:
                    dot = self.create_graph_from_instruction(self.last_analysis)
                    return self.convert_to_base64(dot)
                return self._create_default_graph_image()
                
            self.last_analysis = analysis
            logger.info(f"Meeting analysis result: {analysis}")
            
            # 生成图表指令
            graph_instruction = self.generate_graph_instruction(analysis)
            if not graph_instruction and self.last_analysis:
                graph_instruction = self.last_analysis
                
            # 根据指令创建图表
            dot = self.create_graph_from_instruction(graph_instruction)
            if not dot and self.last_analysis:
                dot = self.create_graph_from_instruction(self.last_analysis)
                
            if not dot:
                return self._create_default_graph_image()
                
            # 转换为base64
            return self.convert_to_base64(dot)
            
        except Exception as e:
            logger.error(f"Error in graph generation: {str(e)}", exc_info=True)
            if self.last_analysis:
                try:
                    dot = self.create_graph_from_instruction(self.last_analysis)
                    return self.convert_to_base64(dot)
                except:
                    pass
            return self._create_default_graph_image()

    def _create_default_graph_image(self):
        """创建默认图表的base64图片"""
        dot = self._create_default_graph()
        return self.convert_to_base64(dot)

    def convert_to_base64(self, dot):
        """将图表转换为base64编码"""
        try:
            png_data = dot.pipe(format='png')
            base64_str = base64.b64encode(png_data).decode('utf-8')
            return f"data:image/png;base64,{base64_str}"
        except Exception as e:
            logger.error(f"Error converting to base64: {str(e)}")
            return None

    def should_generate_graph(self, text_length):
        """检查是否需要生成新图表"""
        self.current_char_count += text_length
        if self.current_char_count >= self.CHAR_THRESHOLD:
            self.current_char_count = 0  # 重置计数器
            return True
        return False