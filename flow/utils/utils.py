import torch
from sklearn.metrics import classification_report
import numpy as np
import os


def get_class_report(model, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

    with torch.no_grad():
        model.eval()
        model.to(device)
        pred = []
        target = []
        for x, y in model.val_dataloader():
            x = x.to(device)
            pred.append(model(x).cpu().numpy())
            target.append(y.numpy())
    pred = np.concatenate(pred)
    target = np.concatenate(target)
    out = classification_report(target, pred.argmax(axis=1))
    print(out)


def readme(path, string):
    with open(os.path.join(path, 'readme.txt'), 'w') as f:
        f.write(string)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
