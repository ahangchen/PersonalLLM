from fastapi import FastAPI
from ds_util import DeepseekUtil

from pydantic import BaseModel

class Query(BaseModel):
    prompt: str
    max_length: int = 1024

app = FastAPI()
ds_util = DeepseekUtil()


@app.post("/generate")
async def generate_text(query: Query):
    answer = ds_util.answer_question(query.prompt, query.max_length)
    return {"response": answer}