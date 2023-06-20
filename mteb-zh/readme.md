# MTEB-zh

## 介绍

MTEB-zh 是一个使用 [MTEB](https://github.com/embeddings-benchmark/mteb) 框架评测中文 Embedding 模型的 BenchMark，包含文本分类，文本重排，以及文本检索等任务。

## 已支持的模型

- [x] [M3E](https://huggingface.co/moka-ai/m3e-base) (m3e-small, m3e-base)
- [x] [text2vec](https://github.com/shibing624/text2vec)
- [x] [DMetaSoul](https://huggingface.co/DMetaSoul/sbert-chinese-general-v2)
- [x] [UER](https://huggingface.co/uer/sbert-base-chinese-nli)
- [x] [ErLangShen](https://huggingface.co/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese)
- [x] [openai](https://openai.com/blog/new-and-improved-embedding-model)

## 评测

### 文本分类

- 数据集选择，选择开源在 HuggingFace 上的 6 种文本分类数据集，包括新闻、电商评论、股票评论、长文本等
- 评测方式，使用 MTEB 的方式进行评测，报告 Accuracy。

|                   | text2vec | m3e-small | m3e-base | openai | DMetaSoul   | uer     | erlangshen  |
| ----------------- | -------- | --------- | -------- | ------ | ----------- | ------- | ----------- |
| TNews             | 0.43     | 0.4443    | **0.4827**   | 0.4594 | 0.3084      | 0.3539  | 0.4361      |
| JDIphone          | 0.8214   | 0.8293    | **0.8533**   | 0.746  | 0.7972      | 0.8283  | 0.8356      |
| GubaEastmony      | 0.7472   | 0.712     | 0.7621   | 0.7574 | 0.735       | 0.7534  | **0.7787**      |
| TYQSentiment      | 0.6099   | 0.6596    | **0.7188**   | 0.68   | 0.6437      | 0.6662  | 0.6444      |
| StockComSentiment | 0.4307   | 0.4291    | 0.4363   | **0.4819** | 0.4309      | 0.4555  | 0.4482      |
| IFlyTek           | 0.414    | 0.4263    | 0.4409   | **0.4486** | 0.3969      | 0.3762  | 0.4241      |
| Average           | 0.5755   | 0.5834    | **0.6157**   | 0.5956 | 0.552016667 | 0.57225 | 0.594516667 |

### 检索排序

#### T2Ranking 1W

- 数据集选择，使用 [T2Ranking](https://github.com/THUIR/T2Ranking/tree/main) 数据集，由于 T2Ranking 的数据集太大，openai 评测起来的时间成本和 api 费用有些高，所以我们只选择了 T2Ranking 中的前 10000 篇文章
- 评测方式，使用 MTEB 的方式进行评测，报告 map@1, map@10, mrr@1, mrr@10, ndcg@1, ndcg@10
- 注意！从实验结果和训练方式来看，除了 M3E 模型和 openai 模型外，其余模型都没有做检索任务的训练，所以结果仅供参考。

|         | text2vec | openai-ada-002 | m3e-small | m3e-base | DMetaSoul | uer     | erlangshen |
| ------- | -------- | -------------- | --------- | -------- | --------- | ------- | ---------- |
| map@1   | 0.4684   | 0.6133         | 0.5574    | **0.626**    | 0.25203   | 0.08647 | 0.25394    |
| map@10  | 0.5877   | 0.7423         | 0.6878    | **0.7656**   | 0.33312   | 0.13008 | 0.34714    |
| mrr@1   | 0.5345   | 0.6931         | 0.6324    | **0.7047**   | 0.29258   | 0.10067 | 0.29447    |
| mrr@10  | 0.6217   | 0.7668         | 0.712     | **0.7841**   | 0.36287   | 0.14516 | 0.3751     |
| ndcg@1  | 0.5207   | 0.6764         | 0.6159    | **0.6881**   | 0.28358   | 0.09748 | 0.28578    |
| ndcg@10 | 0.6346   | 0.7786         | 0.7262    | **0.8004**   | 0.37468   | 0.15783 | 0.39329    |

### 任务介绍

TODO

## 评测已支持的模型

1. 安装依赖
```bash
pip install -r requirements.txt
```
2. 运行评测脚本
```bash
# model_type: m3e_base, erlangshen, uer, d_meta_soul, openai, text2vec ...
python run_mteb_zh <model_type> <model_name: Optional>
```
3. 查看帮助
```bash
python run_mteb_zh --help
```

### 示例

评测 M3E-base 模型
```bash
python run_mteb_zh sentence-transformer moka-ai/m3e-base
```

评测 UER 模型
```bash
python run_mteb_zh uer uer/sbert-base-chinese-nli
```

评测 ErLangShen 模型
```bash
python run_mteb_zh erlangshen
```
case ModelType.m3e_small:
    return SentenceTransformer('moka-ai/m3e-small')
case ModelType.m3e_base:
    return SentenceTransformer('moka-ai/m3e-base')
case ModelType.d_meta_soul:
    return SentenceTransformer('DMetaSoul/sbert-chinese-general-v2')
case ModelType.uer:
    return SentenceTransformer('uer/sbert-base-chinese-nli')

## 评测你的模型

如果你想要将自己的模型也加入到支持列表中，可以直接在此项目中提 issue，我们会第一时间支持您的模型。

```python
from mteb import MTEB
from tasks import TNews

class MyModel():
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        pass

model = MyModel()
evaluation = MTEB(tasks=[TNews()])
evaluation.run(model)
```
