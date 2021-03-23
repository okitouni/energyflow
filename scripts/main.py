from yaml import safe_load
import fire
from scripts.train import train


def main(yaml_file, **kwargs):
    with open(yaml_file, 'r') as f:
        yaml_kwargs = safe_load(f)
    for key, val in kwargs:
        yaml_kwargs[key] = val
    train(**yaml_kwargs)


if __name__ == '__main__':
    fire.Fire(main)
