
import yaml
def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prompt_map = {}

    for name, cfg in data.items():
        
        prompt_map[name] = cfg["content"]

    return prompt_map
