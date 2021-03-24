from yaml import safe_load
import fire
from train import train


def main(yaml, root='~/projects/Geometric-HEP/pythia-gen/data/EIC/', **kwargs):
    with open(yaml, 'r') as f:
        yaml_kwargs = safe_load(f)
    for key, val in kwargs:
        yaml_kwargs[key] = val
    train(root=root, **yaml_kwargs)


if __name__ == '__main__':
    fire.Fire(main)
