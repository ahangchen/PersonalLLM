# 使用Hugging Face Transformers库
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from vector_db import VectorDB
import torch
import time

class DeepseekUtil:
    def __init__(self):
        # self.vector_db = VectorDB()
        # 从Hugging Face下载（需申请访问权限）
        self.model = AutoModelForCausalLM.from_pretrained(
            "./deepseek-14b/deepseek-ai/deepseek-r1-distill-qwen-14b",
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory={0: "10GiB", 1: "10GiB"},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            ),
            model_type="llama",  
        )
        # self.model = self.model.to(torch.float8) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./deepseek-14b/deepseek-ai/deepseek-r1-distill-qwen-14b",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
                )
            )
        
    def __del__(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    def answer_question(self, question, offline=True, max_length=2048):
        if offline:
            prompt = question
        else:
            context = "" # self.vector_db.hybrid_search(question)
            prompt = f"""基于以下论文片段：
            {context}
            
            请用中文专业地回答：{question}"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9
            )
        answer = self.tokenizer.decode(outputs[0])
        return answer
    
def interactive_answer():
    ds_util = DeepseekUtil()
    while(True):
        question = input()
        start_time = time.time() * 1000
        answer = ds_util.answer_question(question)
        end_time = time.time() * 1000
        cost_time = end_time - start_time
        print(answer)
        print('cost', cost_time, 'ms')

    
if __name__ == "__main__":
    interactive_answer()