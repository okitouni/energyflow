from yaml import safe_load, safe_dump
import fire
from train import train
from os.path import join


def main(yaml, root='~/projects/Geometric-HEP/pythia-gen/data/EIC/', **kwargs):
    with open(yaml, 'r') as f:
        yaml_kwargs = safe_load(f)
    for key, val in kwargs:
        yaml_kwargs[key] = val
    out = train(root=root, **yaml_kwargs)

    if yaml_kwargs['logging']:
        with open(join(out, 'config.yaml'), 'w') as f:
            safe_dump(yaml_kwargs, f)


if __name__ == '__main__':
    fire.Fire(main)
