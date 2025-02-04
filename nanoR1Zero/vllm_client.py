import asyncio
import aiohttp
from typing import List, Dict, Any
from dataclasses import dataclass
import math
import time
import subprocess
import requests

@dataclass
class GenerateArgs:
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    number_responses: int = 4

async def process_single_worker(
    worker_id: int,
    worker_url: str,
    sub_dataset: List[Dict],
    batch_size: int,
    args: GenerateArgs
) -> List[Dict[str, Any]]:
    """
    单个worker的处理函数，处理分配给它的子数据集
    Args:
        worker_url: worker的URL
        sub_dataset: 分配给该worker的数据子集
        batch_size: 批处理大小
        args: 生成参数
    """
    print (f"Start worker {worker_id}: processing {len(sub_dataset)} samples")
    t = time.time()
    from tqdm.asyncio import tqdm
    async with aiohttp.ClientSession() as session:
        all_results = []
        total_batches = len(sub_dataset) // batch_size
        
        async for i in tqdm(
            range(0, len(sub_dataset), batch_size),
            total=total_batches,
            desc=f"Worker {worker_id}",  # 只显示URL中的主机名部分
            leave=True
        ):
            batch = sub_dataset[i:i+batch_size]
            try:
                async with session.post(
                    worker_url,
                    json={
                        'prompts': batch,
                        'max_tokens': args.max_tokens,
                        'temperature': args.temperature,
                        'top_p': args.top_p,
                        'number_responses': args.number_responses
                    }
                ) as response:
                    result = await response.json()
                    all_results.extend(result['results'])
                    # print(f"Worker {worker_id} processed batch {i//batch_size + 1}/{math.ceil(len(sub_dataset)/batch_size)}")
            except Exception as e:
                print(f"Error in worker {worker_id} for batch {i//batch_size}: {str(e)}")
                continue
                
        print (f"End worker {worker_id}: processed {len(all_results)} samples, time: {time.time() - t}\n\n")
        return all_results

async def async_batch_generate(
    worker_urls: List[str],
    dataset: List[Dict],
    batch_size: int,
    args: GenerateArgs
) -> List[Dict[str, Any]]:
    """
    并发调用多个worker处理数据集
    
    Args:
        worker_urls: worker URL列表
        dataset: 完整数据集
        batch_size: 批处理大小
        args: 生成参数
    """
    # 计算每个worker应处理的数据量
    worker_count = len(worker_urls)
    items_per_worker = len(dataset) // worker_count
    
    # 划分数据集
    worker_datasets = [
        dataset[i:i+items_per_worker] 
        for i in range(0, len(dataset), items_per_worker)
    ]
    
    # 创建worker任务
    tasks = []
    for worker_idx, worker_dataset in enumerate(worker_datasets):
        if not worker_dataset:  # 跳过空的数据集
            continue
        url = worker_urls[worker_idx]
        task = process_single_worker(worker_idx, url, worker_dataset, batch_size, args)
        tasks.append(task)
    
    # 并发执行所有worker任务
    results = await asyncio.gather(*tasks)
    
    # 合并所有结果
    all_results = []
    for worker_result in results:
        all_results.extend(worker_result)
    
    return all_results

def batch_generate(
    worker_urls: List[str],
    dataset: List[Dict],
    batch_size: int,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    同步版本的批量生成函数
    
    Args:
        worker_urls: worker URL列表
        dataset: 完整数据集
        batch_size: 批处理大小
        **kwargs: 其他生成参数
    """
    args = GenerateArgs(**kwargs)
    return asyncio.run(async_batch_generate(worker_urls, dataset, batch_size, args))


class VLLMClient(object):
    def __init__(self, path: str, ngpus: List[int]):
        self.path = path
        self.ngpus = ngpus
        self.ports = [8000 + i for i in range(len(ngpus))]
        self.worker_urls = [f'http://localhost:{port}' for port in self.ports]
        self.endpoints = [f'{url}/generate_batch' for url in self.worker_urls]
        self.ping_endpoints = [f'{url}/ping' for url in self.worker_urls]
        self.stop_endpoints = [f'{url}/stop' for url in self.worker_urls]
        self.vllm_process = []

    def start_vllm_server(self):
        for device, port in zip(self.ngpus, self.ports):
            log_file = open(f'vllm_server_{device}.log', 'w')
            self.vllm_process.append(subprocess.Popen(['python', './nanoR1Zero/vllm_server.py', self.path, str(device), str(port)], stdout=log_file, stderr=log_file))

        for i in range(30):
            if self.detect_vllm_server():
                print ("vllm_server started")
                return
            time.sleep(10)

    def stop_vllm_server(self):
        try:
            for process in self.vllm_process:
                process.kill()
        except:
            pass

        try:
            for stop_endpoint in self.stop_endpoints:
                requests.get(stop_endpoint)
        except:
            pass
        self.vllm_process = []

    def detect_vllm_server(self):
        is_ready = True
        for ping_endpoint in self.ping_endpoints:
            try:
                response = requests.get(ping_endpoint)
                print (f'{ping_endpoint} status: {response.json()}')
                if response.status_code == 200 and response.json()['status'] == 'ok':
                    pass
                else:
                    is_ready = False
            except Exception as e:
                print (f'{ping_endpoint} error: {e}')
                is_ready = False
        return is_ready

    def generate_batch(self, prompts: List[Dict], batch_size: int, **kwargs):
        args = GenerateArgs(**kwargs)
        return asyncio.run(async_batch_generate(self.endpoints, prompts, batch_size, args))