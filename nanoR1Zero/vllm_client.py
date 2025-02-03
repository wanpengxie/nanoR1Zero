import asyncio
import aiohttp
from typing import List, Dict, Any
from dataclasses import dataclass
import math
import time
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
                
        print (f"End worker {worker_id}: processed {len(all_results)} samples, time: {time.time() - t}")
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

