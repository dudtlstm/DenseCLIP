import json, os

with open('data/BTCV/dataset.json') as f:
    data = json.load(f)

os.makedirs('data/BTCV', exist_ok=True)

with open('data/BTCV/train.txt', 'w') as f:
    for entry in data['training']:
        base = os.path.basename(entry['image']).replace('.png', '')
        f.write(base + '\n')

with open('data/BTCV/val.txt', 'w') as f:
    for entry in data['test']:
        base = os.path.basename(entry['image']).replace('.png', '')
        f.write(base + '\n')