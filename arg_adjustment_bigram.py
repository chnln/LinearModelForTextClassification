import os

for n_bigram in [0, 100, 500, 1000, 5000, 10000]:
    os.system(f"python tc_bigram.py --dataset SST5 --n_classes 5 --n_bigram {n_bigram}")
