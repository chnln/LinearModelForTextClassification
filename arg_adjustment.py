import os

for dataset in ["20news", "SST5"]:
    n_classes = 20 if dataset == "20news" else 5
    for n_features in [200, 2000, 10000]:
        for lr in [0.001, 0.01]:
            for batch_size in [1, 200, 1000]:
                for alpha in [0, 1e-4, 1e-2, 1e-1]:  # 2*3*2*3*4 = 144
                    os.system(f'python text_classification.py'
                          f' --dataset {dataset}'
                          f' --n_classes {n_classes}'
                          f' --n_features {n_features}'
                          f' --lr {lr}'
                          f' --batch_size {batch_size}'
                          f' --alpha {alpha}')
