# nanoR1Zero

最小版本复现deepseek R1 Zero
1. Base model： qwen-1.5b-instruct
2. Prompt: 8k math problems
3. Eval: Math 500

## 训练

```bash
python script/main.py script/conf.json
```


![eval mean rewards](./assets/eval_mean_reward.png)
![eval response length](./assets/eval_response_length.png)
