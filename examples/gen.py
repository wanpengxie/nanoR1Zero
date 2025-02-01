from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F


model_name = "Qwen/Qwen2.5-7B-Instruct"
model_path = '/hy-tmp/Qwen2.5-1.5B-Instruct/'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = "Give me a short introduction to large language model."
messages = [
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


def simple_gen(  
    model,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    pad_token_id=None,
    eos_token_id=None
):
    """
    简单的文本生成函数
    
    参数:
        model: 语言模型
        input_ids: 输入token ids
        max_new_tokens: 最大生成的新token数量
        temperature: 温度参数，控制采样随机性
        top_k: 只保留概率最高的k个token
        top_p: 累积概率阈值
        pad_token_id: padding token的id
        eos_token_id: 结束符token的id
    """
    
    # 用于存储生成的序列
    generated = input_ids
    response = []
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 获取模型输出
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 计算概率分布
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 应用top_k筛选（可选）
            if top_k > 0:
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 应用top_p筛选（可选）
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 移除累积概率超过阈值的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 将新token添加到生成序列中
            # generated = torch.cat([generated, next_token], dim=-1)
            response.append(next_token)
            # 检查是否生成了结束符
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
    
    return torch.tensor([response])


def simple_cot_gen(  
    model,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    pad_token_id=None,
    eos_token_id=None
):
    """
    简单的文本生成函数
    
    参数:
        model: 语言模型
        input_ids: 输入token ids
        max_new_tokens: 最大生成的新token数量
        temperature: 温度参数，控制采样随机性
        top_k: 只保留概率最高的k个token
        top_p: 累积概率阈值
        pad_token_id: padding token的id
        eos_token_id: 结束符token的id
    """
    
    # 用于存储生成的序列
    generated = input_ids
    response = []
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 获取模型输出
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 计算概率分布
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 应用top_k筛选（可选）
            if top_k > 0:
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 应用top_p筛选（可选）
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 移除累积概率超过阈值的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 将新token添加到生成序列中
            generated = torch.cat([generated, next_token], dim=-1)
            response.append(next_token)
            # 检查是否生成了结束符
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
    
    return torch.tensor([response])