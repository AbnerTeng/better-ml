# better-ml

Course materials for TMBA 25th FW

## Usage

**Build ML venv**

```bash
conda create --name <your_venv> python=3.10
conda activate <your_venv>

pip3 install -r requirements.txt
```

**Build PyTorch venv**

Find suitable PyTorch version from [official website](https://pytorch.org/get-started/previous-versions/)

```bash
cd pytorch_tutorial

conda create --name <your_venv> python=3.10
conda activate <your_venv>

pip3 install torch==1.11.0 ## MacOS version here

pip3 install -r requirements.txt
```

Example

```bash
cd pytorch_tutorial

conda create --name pyt_env python=3.10
conda activate pyt_env

pip install torch==1.11.0
pip3 install -r requirements.txt
```

**Run Scripts**

Args

- `lab` (default as `scratch_id3`)
  - `scratch_id3`: Test the ID3 algorithm from pure `numpy`
  - `sklearn_module`: Run the better-sklearn module

Example

```bash
python3 main.py --lab sklearn_module
```

### Execute PyTorch Tutorial

```bash
python -m src.expr.reg_expr --conf_path config/reg_config.yaml
```

### Configs

All the configurations and model settings are stored in `config` directory.
