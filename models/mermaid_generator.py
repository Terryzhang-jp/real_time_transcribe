from langchain_openai import ChatOpenAI
import time
import requests
from base64 import b64encode
import os

class MermaidGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")
        self.retry_count = 0
        self.last_processed_text = ""
        
    def generate_mermaid_prompt(self, text):
        return f"""
        基于以下会议内容，生成一个Mermaid流程图或思维导图。
        请确保图表简洁清晰地展示主要内容和逻辑关系。
        仅返回Mermaid代码，不需要其他解释。
        
        会议内容：
        {text}
        """
    
    def extract_mermaid_code(self, response):
        try:
            code = response.replace("```mermaid", "").replace("```", "").strip()
            return code
        except Exception as e:
            print(f"提取Mermaid代码失败: {e}")
            return None
            
    def convert_to_image_url(self, mermaid_code):
        try:
            encoded = b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
            return f"https://mermaid.ink/img/{encoded}"
        except Exception as e:
            print(f"转换图片URL失败: {e}")
            return None
    
    def process_text(self, current_text, chart_dir):
        if current_text == self.last_processed_text:
            return None
            
        try:
            prompt = self.generate_mermaid_prompt(current_text)
            response = self.llm.predict(prompt)
            
            mermaid_code = self.extract_mermaid_code(response)
            if not mermaid_code:
                raise Exception("Mermaid代码生成失败")
                
            image_url = self.convert_to_image_url(mermaid_code)
            if not image_url:
                raise Exception("图片URL生成失败")
            
            # 下载图片并保存到本地
            response = requests.get(image_url)
            if response.status_code == 200:
                filename = f"chart_{int(time.time())}.png"
                filepath = os.path.join(chart_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                self.last_processed_text = current_text
                self.retry_count = 0
                return filepath
            else:
                raise Exception("图片下载失败")
            
        except Exception as e:
            self.retry_count += 1
            if self.retry_count < 3:
                time.sleep(1)
                return self.process_text(current_text, chart_dir)
            else:
                self.retry_count = 0
                return None
