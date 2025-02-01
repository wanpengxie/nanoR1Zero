from vllm import LLM, SamplingParams
from flask import request, Flask, jsonify
import os
import threading
app = Flask(__name__)
model = {}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']
    model_path = data['model_path']
    max_tokens = data.get('max_len', 8192)
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 1.0)
    num_return_sequences = data.get('number_responses', 16)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        n=num_return_sequences,
    )
    outputs = model['llm'].generate([prompt], sampling_params)
    prompt_token_ids = outputs[0].prompt_token_ids
    output_list = [list(x.token_ids) for x in outputs[0].outputs if x.finish_reason == 'stop']
    output_text = [x.text for x in outputs[0].outputs if x.finish_reason == 'stop']
    return jsonify({
        'prompt_token_ids': prompt_token_ids,
        'output_list': output_list,
        'output_text': output_text
    })


@app.route('/ping', methods=['GET'])
def ping():
    if model.get('status', 'loading') == 'running':
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'loading'})


@app.route('/stop', methods=['GET'])
def stop():
    # exist main process
    os._exit(0)

if __name__ == '__main__':
    from sys import argv
    model_path = argv[1]
    # 非阻塞的启动app
    worker = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8000})
    worker.start()

    model['llm'] = LLM(
        model=model_path,
        tensor_parallel_size=1,    # GPU数量
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",         # 可选 "float16", "bfloat16", "float32"
    )
    model['status'] = 'running'
    worker.join()