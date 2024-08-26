import yaml


def read_yaml_config(config_path: str) -> dict:
    """
    Read a yaml file and return the dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # ------------ If no measures provided we default to all measures ------------ #
    return config
