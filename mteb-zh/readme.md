# MTEB-zh

## 介绍

MTEB-zh 是一个使用 [MTEB](https://github.com/embeddings-benchmark/mteb) 框架评测中文 Embedding 模型的 BenchMark，包含文本分类，文本重排，以及文本检索等任务。

TODO: 添加当前评测

## 任务

TODO: 添加任务介绍

## 评测你的模型

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