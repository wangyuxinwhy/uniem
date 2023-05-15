# uniem
unified embedding model


## Install

```
pip install uniem
```

## Usage

```python
from uniem import UniEmbedder

embedder = UniEmbedder.from_pretrained('uniem/base-softmax-last-mean')
embeddings = embedder.encode(['Hello World!', '你好,世界!'])
```

## Train Your Model

1. create virtual environment
```bash
conda create -n uniem python=3.10
```
2. install uniem
```bash
pip install -e .
```
3. get help
![](./docs/imgs/medi-help.png)
```bash
python scripts/train_medi.py --help
```
4. train embedding model
```bash
python scripts/train_medi.py <model_path_or_name> <data_file>
```
