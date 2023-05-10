# uniem
unified embedding model

## Usage

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
