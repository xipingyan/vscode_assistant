source python-env/bin/activate
# pip install optimum[openvino,nncf] transformers

model_id=Qwen/Qwen2.5-Coder-7B-Instruct/

optimum-cli export openvino --model $model_id --weight-format int4 --task text-generation  ./ov_model/