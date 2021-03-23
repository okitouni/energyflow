from yaml import safe_load
import fire
from scripts.train import train


def main(yaml_file):
    with open(yaml_file, 'r') as f:
        kwargs = safe_load(f)
    train(**kwargs)


if __name__ == '__main__':
    fire.Fire(main)
