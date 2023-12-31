import json

from typing import Dict, Any
from pathlib import Path
from jsonschema import validate, ValidationError


# pylint: disable = use-dict-literal

def read_json(path):
    file_path = Path(path)
    if not file_path.is_file():
        raise AssertionError(f"Couldn't open file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as config_file:
        return json.load(config_file)


DEFAULT_SCHEMA_PATH = Path(Path(__file__).resolve().parent.parent) / 'schemas/model_config_schema.json'
DEFAULT_SCHEMA = read_json(DEFAULT_SCHEMA_PATH)


class ConfigException(Exception):
    """
    Raise when we cannot parse arguments or open json file
    """


class ModelConfig:
    """
    Singleton class that stores application settings, defined in json file, as dictionary
    """

    _instance = None
    _properties: Dict[str, Any] = dict()
    _default_file_path = Path(__file__).resolve().parent / "samples" / "default.json"

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        if not ModelConfig._instance:
            ModelConfig._instance = super(ModelConfig, cls).__new__(cls)
        return ModelConfig._instance

    def __init__(self, file_path=None):
        if ModelConfig._properties:
            return

        self._file_path = file_path or ModelConfig._default_file_path
        self._json_data = self._load_from_json()
        ModelConfig._properties = {}

        for name, value in self._json_data.items():
            ModelConfig._properties[name] = value

    def _load_from_json(self) -> dict:
        try:
            with open(self._file_path, encoding="utf-8") as cfg_file:
                json_conf = json.load(cfg_file)
                validate(json_conf, DEFAULT_SCHEMA)
                return json_conf
        except IOError as err:
            raise ConfigException(f"Cannot open file '{self._file_path}'") from err
        except ValidationError as err:
            raise ConfigException("File does not match default json schema") from err

    @property
    def properties(self) -> dict:
        """
        :return: config properties as a dictionary
        """
        return self._properties
