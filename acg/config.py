import yaml

class Config:
    def __init__(self, config_path="params.yml"):
        self._config = self._load_config(config_path)

    def _load_config(self, path):
        """Load YAML configuration file."""
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def __getattr__(self, item):
        """Enable dot notation access for top-level and nested keys."""
        value = self._config.get(item)
        if isinstance(value, dict):
            # Wrap nested dictionaries in Config for recursive dot notation access
            return Config._dict_to_obj(value)
        elif value is not None:
            return value
        raise AttributeError(f"Configuration key '{item}' not found.")

    @staticmethod
    def _dict_to_obj(data):
        """Recursively convert a dictionary into an object supporting dot notation."""
        if not isinstance(data, dict):
            return data
        return type("DotDict", (object,), {k: Config._dict_to_obj(v) for k, v in data.items()})()


config = Config("../params.yml")
