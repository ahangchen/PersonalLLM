import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ds_util import DeepseekUtil
from contextlib import asynccontextmanager

# 1. 定义模型和分词器的全局变量（异步生命周期管理）
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 加载模型和分词器
    model_dir = "./deepseek-33b"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    yield
    # 清理资源（可选）
    del model
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# 2. 定义请求体格式（兼容 OpenAI ChatCompletion）
class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

# 3. 定义 API 端点
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    try:
        # 提取用户输入
        user_input = request.messages[-1]["content"]
        
        # 使用分词器编码输入
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        
        # 生成文本
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
        
        # 解码输出
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 构造兼容 OpenAI 的响应
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                }
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))