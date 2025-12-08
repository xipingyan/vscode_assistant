# pip install accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

model_name = "./Qwen/Qwen2.5-Coder-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa" # "eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
    write a quick sort algorithm. language: c/cpp, range:[0, n], input vector: float vec[n]. please answer follow json format, for example:
    "{
        `code`: "here is source code",
        `comment`: "// describle this function"
    }"
    do not generate other code.
    """

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"response={response}")

match = re.search(r'\{[\s\S]*\}', response)
clean_json_string = None
if match:
    clean_json_string = match.group(0)
    try:
        response_dict = json.loads(clean_json_string)
        print(f"code=\n{response_dict['code']}")
        print(f"comment=\n{response_dict['comment']}")
    except json.JSONDecodeError as e:
        print(f"❌ Internal JSON block decode error: {e}")
else:
    print("⚠️ Can find valid JSON.")