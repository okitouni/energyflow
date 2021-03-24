import os

os.chdir('/home/kitouni/projects/Geometric-HEP/pythia-gen')
from yaml import safe_load
import fire
from train import train


def main(root, yaml_file, **kwargs):
    with open(yaml_file, 'r') as f:
        yaml_kwargs = safe_load(f)
    for key, val in kwargs:
        yaml_kwargs[key] = val
    train(root=root, **yaml_kwargs)


if __name__ == '__main__':
    fire.Fire(main)
