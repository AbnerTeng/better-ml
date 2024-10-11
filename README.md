# better-ml

Course materials for TMBA 25th FW

## Usage

**Build venv**

```bash
conda create --name <your_venv> python=3.10
conda activate <your_venv>

pip3 install -r requirements.txt
```

Example

```bash
conda create --name temp_env python=3.10
conda activate temp_env

pip3 install -r requirements.txt
```

**Run Scripts**

Args

- `lab` (default as `scratch_id3`)
    - `scratch_id3`: Test the ID3 algorithm from pure `numpy`
    - `sklearn_module`: Run the better-sklearn module


Example
```
python3 main.py --lab sklearn_module
```
