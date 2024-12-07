from typing import Any, Dict
import json

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    load model config
    """
    file_type = path.split(".")[-1]

    if file_type == "json":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config

    elif file_type in ["yaml", "yml"]:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config
