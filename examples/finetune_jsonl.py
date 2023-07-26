import pandas as pd
from accelerate import DistributedDataParallelKwargs
from uniem.finetuner import FineTuner

# 读取 jsonl 文件
df = pd.read_json('example_data/riddle.jsonl', lines=True)
# 重新命名
df = df.rename(columns={'instruction': 'text', 'output': 'text_pos'})
# 指定训练的模型为 m3e-small
finetuner = FineTuner.from_pretrained('moka-ai/m3e-small', dataset=df.to_dict('records'))
finetuner.run(
    epochs=1, 
    output_dir='finetuned-model-riddle',
    batch_size=32,
    accelerator_kwargs={
        'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters=True)]
    }
)
