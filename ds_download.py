from modelscope import snapshot_download

model_dir = snapshot_download('deepseek-ai/deepseek-r1-distill-qwen-14b', cache_dir='./deepseek-14b')