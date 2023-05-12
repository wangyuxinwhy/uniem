# uniem
unified embedding model

# Usage

```python
from uniem import UniEmbedder


sentences = ['Hello, World!', '你好，世界！']
embedder = UniEmbedder()
embeddings = embedder(sentences)
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
```bash
python scripts/train_medi.py --help
```
4. train embedding model
```bash
python scripts/train_medi.py <model_path_or_name> <data_file>
```
