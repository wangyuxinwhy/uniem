# uniem

uniem 项目的目标是创建中文最好的通用文本嵌入模型。

本项目主要包括模型的训练，微调和评测代码，模型与数据集会在 [HuggingFace](https://huggingface.co/) 社区上进行开源。

## 🌟 重要更新

- 🎉 **2023.06.08** , 发布 [M3E models](https://huggingface.co/moka-ai/m3e-base) ，在中文文本分类和文本检索上均优于 `openai text-embedding-ada-002`，详请请参考 [M3E models README](https://huggingface.co/moka-ai/m3e-base/blob/main/README.md)。
- 📊 **2023.06.17** , 发布 [MTEB-zh](https://github.com/wangyuxinwhy/uniem/tree/main/mteb-zh) 正式版 ， 支持 6 大类 Embedding 模型 ，支持 4 大类任务 ，共 9 种数据集的自动化评测
- ➿ **2023.06.17** , 发布 uniem 0.2.1 ， 实现了 `FineTuner` 以原生支持模型微调，**几行代码，即刻适配**！

## 🔧 使用 M3E

M3E 系列模型完全兼容 [sentence-transformers](https://www.sbert.net/) ，你可以通过 **替换模型名称** 的方式在所有支持 sentence-transformers 的项目中无缝使用 M3E Models，比如 [chroma](https://docs.trychroma.com/getting-started), [guidance](https://github.com/microsoft/guidance), [semantic-kernel](https://github.com/microsoft/semantic-kernel) 。

### 安装

```bash
pip install sentence-transformers
```

### 使用 

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("moka-ai/m3e-base")
embeddings = model.encode(['Hello World!', '你好,世界!'])
```

## 🎨 微调模型

`uniem` 提供了非常易用的 finetune 接口，几行代码，即刻适配！

```python
from datasets import load_dataset

from uniem.finetuner import FineTuner

dataset = load_dataset('shibing624/nli_zh', 'STS-B')
# 指定训练的模型为 m3e-small
finetuner = FineTuner('moka-ai/m3e-small', dataset=dataset)
finetuner.run(epochs=1)
```

微调的模型详见 [uniem 微调教程](https://github.com/wangyuxinwhy/uniem/blob/main/examples/finetune.ipynb) or <a target="_blank" href="https://colab.research.google.com/github/wangyuxinwhy/uniem/blob/main/examples/finetune.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 💯 MTEB-zh

中文 Embedding 模型缺少统一的评测标准，所以我们参考了 [MTEB](https://huggingface.co/spaces/mteb/leaderboard) ，构建了中文评测标准 MTEB-zh，目前已经对 6 种模型在各种数据集上进行了横评，详细的评测结果请参考 [MTEB-zh](https://github.com/wangyuxinwhy/uniem/tree/main/mteb-zh) 。

如果您想要在 MTEB-zh 中添加评测数据集或者模型，欢迎提 issue 或者 PR，我会在第一时间进行支持，期待您的贡献！