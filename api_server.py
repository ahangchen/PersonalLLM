import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ds_util import DeepseekUtil
from contextlib import asynccontextmanager

global deep_seek
# 1. 定义模型和分词器的全局变量（异步生命周期管理）
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 加载模型和分词器
    global deep_seek
    deep_seek = DeepseekUtil()
    yield

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
        global deep_seek
        response_text = deep_seek.answer_question(user_input)
        
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