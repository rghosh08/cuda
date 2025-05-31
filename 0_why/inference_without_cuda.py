import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/inference', methods=['POST'])
def inference():
    prompt = request.json.get('prompt', 'Hello')
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"result": result})
