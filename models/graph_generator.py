from graphviz import Digraph
import base64
import io
from langchain_openai import ChatOpenAI
import time
import logging

logger = logging.getLogger(__name__)

class GraphvizGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")
        self.last_processed_text = ""
        self.last_update_time = 0
        self.update_interval = 30  # 30秒更新间隔
        
    def generate_graph_prompt(self, text):
        return f"""
        基于以下会议内容，生成一个逻辑关系图的节点和边的描述。
        格式要求：
        1. 每行表示一个关系
        2. 格式为 "节点A -> 节点B: 关系描述"
        3. 最多生成8-10个关系
        4. 保持简洁明了
        
        会议内容：
        {text}
        """
        
    def parse_relationships(self, llm_response):
        relationships = []
        for line in llm_response.strip().split('\n'):
            if '->' in line:
                source, rest = line.split('->')
                target, label = rest.split(':', 1) if ':' in rest else (rest, '')
                relationships.append({
                    'source': source.strip(),
                    'target': target.strip(),
                    'label': label.strip()
                })
        return relationships
        
    def create_graph(self, relationships):
        dot = Digraph(comment='Logic Flow')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box', style='rounded')
        
        for rel in relationships:
            dot.node(rel['source'])
            dot.node(rel['target'])
            dot.edge(rel['source'], rel['target'], label=rel['label'])
            
        return dot
        
    def convert_to_base64(self, dot):
        png_data = dot.pipe(format='png')
        base64_str = base64.b64encode(png_data).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
        
    def should_update(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
        
    def process_text(self, current_text):
        if not current_text.strip():
            logger.warning("Empty text received")
            return None
            
        try:
            logger.info("Starting graph generation")
            
            # 生成图描述
            prompt = self.generate_graph_prompt(current_text)
            logger.debug(f"Generated prompt: {prompt}")
            
            # 调用 GPT-4
            logger.info("Calling GPT-4...")
            response = self.llm.predict(prompt)
            logger.debug(f"LLM response: {response}")
            
            # 解析关系
            logger.info("Parsing relationships...")
            relationships = self.parse_relationships(response)
            logger.debug(f"Parsed relationships: {relationships}")
            
            if not relationships:
                logger.warning("No relationships found in the response")
                return None
                
            # 创建图
            logger.info("Creating graph...")
            try:
                dot = self.create_graph(relationships)
            except Exception as e:
                logger.error(f"Error creating graph: {e}")
                raise Exception(f"Graphviz error: {e}")
                
            # 转换为base64
            logger.info("Converting to base64...")
            try:
                base64_image = self.convert_to_base64(dot)
            except Exception as e:
                logger.error(f"Error converting to base64: {e}")
                raise Exception(f"Base64 conversion error: {e}")
                
            return base64_image
            
        except Exception as e:
            logger.error(f"Error in graph generation: {str(e)}", exc_info=True)
            raise