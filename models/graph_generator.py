from graphviz import Digraph
import base64
import io
from langchain_openai import ChatOpenAI
import time
import logging
import json
import copy
import os

logger = logging.getLogger(__name__)

class MeetingAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini")
        
    def analyze_meeting(self, text):
        prompt = f"""
        角色：你是一个专业的会议内容分析专家。请分析以下文本，提取主题和逻辑结构。

        文本内容：
        {text}

        请以JSON格式返回分析结果，要求：
        1. 主题（topic）：核心主题
        2. 子节点（children）：可以包含任意数量的子节点，每个子节点可以有：
           - content: 节点内容
           - type: 节点类型（"main"主要论点, "detail"细节, "example"例子, "extension"延伸讨论）
           - children: 该节点的子节点数组（可以继续嵌套）

        输出格式：
        {{
            "topic": "主题内容",
            "children": [
                {{
                    "content": "主要论点1",
                    "type": "main",
                    "children": [
                        {{
                            "content": "支持论据1",
                            "type": "detail",
                            "children": []
                        }},
                        {{
                            "content": "延伸讨论",
                            "type": "extension",
                            "children": [
                                {{
                                    "content": "具体例子",
                                    "type": "example",
                                    "children": []
                                }}
                            ]
                        }}
                    ]
                }}
            ]
        }}

        要求：
        1. 保持JSON格式的合法性
        2. 根据内容的逻辑关系自然延伸，不限制层级深度
        3. 重要的论点可以展开更多层级
        4. 次要的论点可以较少层级
        5. 使用不同的节点类型来表达不同的内容性质
        """
        
        try:
            response = self.llm.invoke(prompt)
            # 从 AIMessage 中提取内容
            content = response.content
            
            # 移除可能的 markdown 标记
            if content.startswith('```json'):
                content = content[7:-3]  # 移除 ```json 和 ```
                
            # 解析 JSON
            analysis = json.loads(content)
            
            if self._validate_analysis(analysis):
                return analysis
            else:
                logger.error("Invalid analysis format")
                return self._get_default_analysis()
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return self._get_default_analysis()

    def _validate_analysis(self, analysis):
        """验证分析结果格式"""
        try:
            if not isinstance(analysis, dict):
                return False
            
            if 'topic' not in analysis or not analysis['topic']:
                return False
                
            if 'children' not in analysis or not isinstance(analysis['children'], list):
                return False
                
            def validate_node(node):
                if not isinstance(node, dict):
                    return False
                if 'content' not in node or not node['content']:
                    return False
                if 'type' not in node or node['type'] not in ['main', 'detail', 'example', 'extension']:
                    return False
                if 'children' not in node or not isinstance(node['children'], list):
                    return False
                return all(validate_node(child) for child in node['children'])
                
            return all(validate_node(child) for child in analysis['children'])
            
        except Exception:
            return False

    def _get_default_analysis(self):
        """返回默认的分析结果"""
        return {
            "topic": "会议进行中",
            "children": [
                {
                    "content": "正在收集要点",
                    "type": "main",
                    "children": []
                }
            ]
        }

class GraphvizGenerator:
    def __init__(self):
        self.analyzer = MeetingAnalyzer()
        self.last_analysis = None
        self.current_char_count = 0
        self.CHAR_THRESHOLD = 150
        # 使用项目中的字体文件
        self.font_name = 'YRDZST Semibold'
        # 获取字体文件的绝对路径
        self.font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'YRDZST Semibold.ttf')

    def generate_graph_instruction(self, analysis):
        """根据分析结果生成图表指令"""
        try:
            nodes = []
            edges = []
            
            # 创建主题节点
            main_id = 'main'
            nodes.append({
                'id': main_id,
                'label': self._format_label(analysis['topic']),
                'style': self._get_node_style('root')
            })
            
            def process_node(node, parent_id, node_index, level):
                """递归处理节点"""
                current_id = f'node_{level}_{node_index}'
                
                # 根据节点类型选择样式
                style = self._get_node_style(node['type'], level)
                
                # 添加节点
                nodes.append({
                    'id': current_id,
                    'label': self._format_label(node['content']),
                    'style': style
                })
                
                # 添加边
                edges.append({
                    'from': parent_id,
                    'to': current_id,
                    'style': self._get_edge_style(node['type'], level)
                })
                
                # 递归处理子节点
                for idx, child in enumerate(node['children']):
                    process_node(child, current_id, idx, level + 1)
            
            # 处理所有子节点
            for idx, child in enumerate(analysis['children']):
                process_node(child, main_id, idx, 1)
            
            return {
                'nodes': nodes,
                'edges': edges,
                'graph_attrs': {
                    'rankdir': 'TB',
                    'splines': 'ortho',
                    'nodesep': '0.5',
                    'ranksep': '0.7',
                    'fontname': 'Microsoft YaHei',
                    'concentrate': 'true'
                }
            }
        except Exception as e:
            logger.error(f"Error generating graph instruction: {str(e)}")
            return None

    def _get_node_style(self, node_type, level=0):
        """根据节点类型和层级返回样式"""
        base_style = {
            'shape': 'box',
            'style': 'filled,rounded',
            'fontsize': str(14 - level) if level < 4 else '10'
        }
        
        colors = {
            'root': 'lightblue',
            'main': 'lightgreen',
            'detail': '#E8E8E8',
            'example': '#FFE4B5',
            'extension': '#E6E6FA'
        }
        
        base_style['fillcolor'] = colors.get(node_type, 'white')
        return base_style

    def _get_edge_style(self, node_type, level):
        """根据节点类型和层级返回边的样式"""
        base_style = {
            'color': 'gray' if level > 2 else 'black',
            'arrowhead': 'normal',
            'penwidth': str(1.5 - level * 0.2) if level < 3 else '0.8'
        }
        
        if node_type in ['example', 'extension']:
            base_style['style'] = 'dashed'
        
        return base_style

    def create_graph_from_instruction(self, instruction):
        """从指令创建图表"""
        try:
            dot = Digraph(comment='Meeting Analysis')
            
            # 设置全局属���
            dot.attr(
                rankdir='TB',
                splines='ortho',
                nodesep='0.5',
                ranksep='0.7',
                fontname=self.font_name
            )
            
            # 添加节点和边
            if instruction.get('nodes'):
                for node in instruction['nodes']:
                    style = {k: str(v) for k, v in node.get('style', {}).items()}
                    style['fontname'] = self.font_name
                    dot.node(str(node['id']), str(node['label']), **style)
            
            if instruction.get('edges'):
                for edge in instruction['edges']:
                    style = {k: str(v) for k, v in edge.get('style', {}).items()}
                    style['fontname'] = self.font_name
                    dot.edge(str(edge['from']), str(edge['to']), '', **style)
            
            # 验证生成的图表
            try:
                png_data = dot.pipe(format='png')
                return dot
            except Exception as e:
                logger.error(f"Graph validation failed: {str(e)}")
                return self._create_default_graph()
            
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
            
        # 添加字符数检查
        if not self.should_generate_graph(len(current_text)):
            # 如果没到150字，返回上一次的图表
            if self.last_analysis:
                dot = self.create_graph_from_instruction(self.last_analysis)
                return self.convert_to_base64(dot)
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

    def _create_default_graph(self):
        """创建默认的图表"""
        try:
            dot = Digraph(comment='Default Graph')
            dot.attr(rankdir='TB')
            dot.attr(fontname=self.font_name)
            
            # 创建默认的主节点
            dot.node('default', '等待分析中...', 
                    shape='box',
                    style='filled,rounded',
                    fillcolor='lightgray',
                    fontsize='12',
                    fontname=self.font_name)
            
            return dot
        except Exception as e:
            logger.error(f"Error creating default graph: {str(e)}")
            return None