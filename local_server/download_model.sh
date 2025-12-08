# pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

model_id=Qwen/Qwen2.5-Coder-7B-Instruct
hf download $model_id --local-dir $model_id --token [token_id] --max-workers 20