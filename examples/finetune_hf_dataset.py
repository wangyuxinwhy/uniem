from datasets import load_dataset
from uniem.finetuner import FineTuner

dataset = load_dataset('shibing624/nli_zh', 'STS-B')
finetuner = FineTuner('moka-ai/m3e-small', dataset=dataset)  # type: ignore
finetuner.run(epochs=1)
