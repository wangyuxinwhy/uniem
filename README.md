# uniem
[![Release](https://img.shields.io/pypi/v/uniem)](https://pypi.org/project/uniem/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/uniem)
[![ci](https://github.com/wangyuxinwhy/uniem/actions/workflows/ci.yml/badge.svg)](https://github.com/wangyuxinwhy/uniem/actions/workflows/ci.yml)
[![cd](https://github.com/wangyuxinwhy/uniem/actions/workflows/cd.yml/badge.svg)](https://github.com/wangyuxinwhy/uniem/actions/workflows/cd.yml)

uniem 项目的目标是创建中文最好的通用文本嵌入模型。

本项目主要包括模型的训练，微调和评测代码，模型与数据集会在 [HuggingFace](https://huggingface.co/) 社区上进行开源。

## 🌟 重要更新

- ➿ **2023.07.11** , 发布 uniem 0.3.0， `FineTuner` 除 M3E 外，还支持 `sentence_transformers`, `text2vec` 等模型的微调，同时还支持 [SGPT](https://github.com/Muennighoff/sgpt) 的方式对 GPT 系列模型进行训练，以及 Prefix Tuning。 **FineTuner 初始化的 API 有小小的变化，无法兼容 0.2.0**
- ➿ **2023.06.17** , 发布 uniem 0.2.1 ， 实现了 `FineTuner` 以原生支持模型微调，**几行代码，即刻适配**！
- 📊 **2023.06.17** , 发布 [MTEB-zh](https://github.com/wangyuxinwhy/uniem/tree/main/mteb-zh) 正式版 ， 支持 6 大类 Embedding 模型 ，支持 4 大类任务 ，共 9 种数据集的自动化评测
- 🎉 **2023.06.08** , 发布 [M3E models](https://huggingface.co/moka-ai/m3e-base) ，在中文文本分类和文本检索上均优于 `openai text-embedding-ada-002`，详请请参考 [M3E models README](https://huggingface.co/moka-ai/m3e-base/blob/main/README.md)。

## 🔧 使用 M3E

M3E 系列模型完全兼容 [sentence-transformers](https://www.sbert.net/) ，你可以通过 **替换模型名称** 的方式在所有支持 sentence-transformers 的项目中无缝使用 M3E Models，比如 [chroma](https://docs.trychroma.com/getting-started), [guidance](https://github.com/microsoft/guidance), [semantic-kernel](https://github.com/microsoft/semantic-kernel) 。

安装

```bash
pip install sentence-transformers
```

使用 

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

中文 Embedding 模型缺少统一的评测标准，所以我们参考了 [MTEB](https://huggingface.co/spaces/mteb/leaderboard) ，构建了中文评测标准 MTEB-zh，目前已经对 6 种模型在各种数据集上进行了横评，详细的评测方式和代码请参考 [MTEB-zh](https://github.com/wangyuxinwhy/uniem/tree/main/mteb-zh) 。


### 文本分类

- 数据集选择，选择开源在 HuggingFace 上的 6 种文本分类数据集，包括新闻、电商评论、股票评论、长文本等
- 评测方式，使用 MTEB 的方式进行评测，报告 Accuracy。

|                   | text2vec | m3e-small | m3e-base | m3e-large-0619 | openai | DMetaSoul   | uer     | erlangshen  |
| ----------------- | -------- | --------- | -------- | ------ | ----------- | ------- | ----------- | ----------- |
| TNews             | 0.43     | 0.4443    | 0.4827   | **0.4866** | 0.4594 | 0.3084      | 0.3539  | 0.4361      |
| JDIphone          | 0.8214   | 0.8293    | 0.8533   | **0.8692** | 0.746  | 0.7972      | 0.8283  | 0.8356      |
| GubaEastmony      | 0.7472   | 0.712     | 0.7621   | 0.7663 | 0.7574 | 0.735       | 0.7534  | **0.7787**      |
| TYQSentiment      | 0.6099   | 0.6596    | 0.7188   | **0.7247** | 0.68   | 0.6437      | 0.6662  | 0.6444      |
| StockComSentiment | 0.4307   | 0.4291    | 0.4363   | 0.4475 | **0.4819** | 0.4309      | 0.4555  | 0.4482      |
| IFlyTek           | 0.414    | 0.4263    | 0.4409   | 0.4445 | **0.4486** | 0.3969      | 0.3762  | 0.4241      |
| Average           | 0.5755   | 0.5834    | 0.6157   | **0.6231** | 0.5956 | 0.552016667 | 0.57225 | 0.594516667 |

### 检索排序

- 数据集选择，使用 [T2Ranking](https://github.com/THUIR/T2Ranking/tree/main) 数据集，由于 T2Ranking 的数据集太大，openai 评测起来的时间成本和 api 费用有些高，所以我们只选择了 T2Ranking 中的前 10000 篇文章
- 评测方式，使用 MTEB 的方式进行评测，报告 map@1, map@10, mrr@1, mrr@10, ndcg@1, ndcg@10

|         | text2vec | openai-ada-002 | m3e-small | m3e-base | m3e-large-0619 | DMetaSoul | uer     | erlangshen |
| ------- | -------- | -------------- | --------- | -------- | --------- | ------- | ---------- | ---------- |
| map@1   | 0.4684   | 0.6133         | 0.5574    | **0.626**    | 0.6256 | 0.25203   | 0.08647 | 0.25394    |
| map@10  | 0.5877   | 0.7423         | 0.6878    | **0.7656**   | 0.7627 | 0.33312   | 0.13008 | 0.34714    |
| mrr@1   | 0.5345   | 0.6931         | 0.6324    | 0.7047   | **0.7063** | 0.29258   | 0.10067 | 0.29447    |
| mrr@10  | 0.6217   | 0.7668         | 0.712     | **0.7841**   | 0.7827 | 0.36287   | 0.14516 | 0.3751     |
| ndcg@1  | 0.5207   | 0.6764         | 0.6159    | 0.6881   | **0.6884** | 0.28358   | 0.09748 | 0.28578    |
| ndcg@10 | 0.6346   | 0.7786         | 0.7262    | **0.8004**   | 0.7974 | 0.37468   | 0.15783 | 0.39329    |

## 🤝 Contributing

如果您想要在 MTEB-zh 中添加评测数据集或者模型，欢迎提 issue 或者 PR，我会在第一时间进行支持，期待您的贡献！

## 📜 License

uniem is licensed under the Apache-2.0 License. See the LICENSE file for more details.