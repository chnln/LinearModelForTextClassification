import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller
import argparse
import random
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import csv


class LogLinearModel:
    def __init__(self, n_classes, n_features, lr, alpha):
        self.n_classes = n_classes
        self.n_features = n_features
        self.lr = lr
        self.alpha = alpha
        self.w = [[0.0 for i in range(n_features)] for k in range(n_classes)]

    def transform(self, samples):
        labels = []
        list_scores = []
        for sample in samples:
            scores = []
            for i_label in range(self.n_classes):
                # calculate score(x, y)
                # score(x, y) = f_i(x, y) * lambda_(f_i(x, y)), i range from (0, n_features)
                scores.append(sum([self.w[i_label][token] for token in sample]))
            scores = np.array(scores)
            scores = np.exp(scores-scores.max())  # math range error/overflow encounted in exp if not minus max()
            scores /= scores.sum()
            label = np.argmax(scores)
            labels.append(label)
            list_scores.append(scores.tolist())
        return labels, list_scores

    def update(self, samples, labels):
        _, list_probs = self.transform(samples)
        gradient = [[0.0 for i in range(self.n_features)] for k in range(self.n_classes)]
        for i_sample, sample in enumerate(samples):
            for i_feature in sample:
                for i_class in range(self.n_classes):
                    if i_class == labels[i_sample]:
                        gradient[i_class][i_feature] += 1.0  # empirical counts
                    # expected counts and regularization
                    gradient[i_class][i_feature] -= (list_probs[i_sample][i_class] + self.alpha * self.w[i_class][i_feature])
        for i_class in range(self.n_classes):
            for i_feature in range(self.n_features):
                # update lambda
                self.w[i_class][i_feature] += self.lr * gradient[i_class][i_feature]
        return

    def fit(self, samples, labels, batch_size):
        count = 0
        for i_batch in tqdm(range((len(samples) // batch_size + 1))):
            batch_samples = samples[count: count + batch_size]
            batch_labels = labels[count: count + batch_size]
            self.update(batch_samples, batch_labels)
            count += batch_size
        return


def get_lemma_dict():
    with open('lemmatization-en.txt', 'r') as f:
        lines = f.readlines()
        res_dict = {}
        for line in lines:
            lemma, word = line.strip().split()
            res_dict[word] = lemma
    return res_dict


def preprocess_doc(doc):
    # remove non-letters and numbers, unnecessary punctuations; transform to lower case
    doc = re.sub(r"[^A-Za-z0-9]", " ", doc).lower()
    # tokenization
    tokens = word_tokenize(doc)

    res = []
    for token in tokens:
        # lemmatization
        t = lemma_dict[token] if token in lemma_dict else token

        # spelling correction -- too slow
        # spell = Speller()
        # t = spell(t)

        # stop-words, length limit
        if t not in stop_words and len(t) >= 3:
            res.append(t)
    return res


def extract_features_bigram(tokens):
    features = []
    for token in tokens:
        if token in bigram2id:
            features.append(bigram2id[token])
    for i in range(len(tokens)-1):
        if (tokens[i], tokens[i+1]) in bigram2id:
            features.append(bigram2id[(tokens[i], tokens[i+1])])
    return features


def get_data_shuffled(features, labels):
    length = len(features)
    merge_list = [[features[i], labels[i]] for i in range(length)]
    random.shuffle(merge_list)
    return [merge_list[i][0] for i in range(length)], [merge_list[i][1] for i in range(length)]


def evaluation(pred, labels):
    evals = dict()
    evals["accuracy"] = accuracy_score(pred, labels)
    evals["F1-macro"] = f1_score(pred, labels, average="macro")
    evals["F1-micro"] = f1_score(pred, labels, average="micro")
    return evals


if __name__ == '__main__':
    fp_sst_train = r"SST5/sst_train.csv"
    fp_sst_test = r"SST5/sst_test.csv"
    fp_20news_train = r"20news/train.csv"
    fp_20news_label = r"20news/label.csv"
    fp_20news_test = r"20news/test.csv"

    parser = argparse.ArgumentParser(description="arguments for a log-linear model")
    # parser.add_argument("--dataset", default="20news", type=str, help="dataset, 20news or SST5")
    # parser.add_argument("--n_classes", default=20, type=int, help="number of classes")
    parser.add_argument("--dataset", default="SST5", type=str, help="dataset, 20news or SST5")
    parser.add_argument("--n_classes", default=5, type=int, help="number of classes")
    parser.add_argument("--n_features", default=10000, type=int, help="number of features")
    parser.add_argument("--batch_size", default=200, type=int, help="size of each batch")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--alpha", default=0.0001, type=float, help="regularization coefficient")
    parser.add_argument("--n_epoches", default=10, type=int, help="number of epoches")
    parser.add_argument("--n_bigram", default=1000, type=int, help="number of bigram features")
    args = parser.parse_args()
    print(args)

    if args.dataset == "20news":
        fp_train = fp_20news_train
        fp_test = fp_20news_test
    else:
        fp_train = fp_sst_train
        fp_test = fp_sst_test
    data_train = pd.read_csv(fp_train)
    data_test = pd.read_csv(fp_test)

    # preprocess
    lemma_dict = get_lemma_dict()
    print("got lemmatization dictionary")
    stop_words = set(stopwords.words("english"))
    print("got stop-word list")

    data_train["tokens"] = data_train["data"].apply(lambda x: preprocess_doc(x))
    data_test["tokens"] = data_test["data"].apply(lambda x: preprocess_doc(x))
    data_train["bigram"] = data_train["tokens"].apply(lambda x: [(x[i], x[i + 1]) for i in range(len(x) - 1)])
    data_test["bigram"] = data_test["tokens"].apply(lambda x: [(x[i], x[i + 1]) for i in range(len(x) - 1)])

    vocabulary = data_train.tokens.apply(pd.Series).stack().value_counts().to_dict()
    vocabulary_bigram = data_train.bigram.apply(pd.Series).stack().value_counts().to_dict()
    print("got vocabulary")

    bigram2id = dict()
    for i, item in enumerate(vocabulary.items()):
        if i >= (args.n_features - args.n_bigram):
            break
        bigram2id[item[0]] = i

    for i, item in enumerate(vocabulary_bigram.items()):
        if i >= args.n_bigram:
            break
        bigram2id[item[0]] = (i + args.n_features - args.n_bigram)

    data_train["features_bigram"] = (data_train["tokens"] + data_train["bigram"]).apply(lambda x: extract_features_bigram(x))
    print("got features of training file")
    data_test["features_bigram"] = (data_test["tokens"] + data_test["bigram"]).apply(lambda x: extract_features_bigram(x))
    print("got features of testing file")

    # id2token = dict([(v, k) for k, v in token2id.items()])
    # data_train["features"].apply(lambda x: [id2token[i] for i in x])

    list_train_features = data_train.features_bigram.to_list()
    list_train_labels = data_train.target.to_list()
    list_test_features = data_test.features_bigram.to_list()
    list_test_labels = data_test.target.to_list()

    model = LogLinearModel(n_classes=args.n_classes, n_features=args.n_features, lr=args.lr, alpha=args.alpha)

    df_evals = pd.DataFrame(columns=["train-acc", "train-f1", "test-acc", "test-f1"])
    for i_epoch in range(args.n_epoches):
        shuffled_f, shuffled_l = get_data_shuffled(list_train_features, list_train_labels)
        model.fit(shuffled_f, shuffled_l, batch_size=args.batch_size)
        evals_train = evaluation(model.transform(list_train_features)[0], list_train_labels)
        evals_test = evaluation(model.transform(list_test_features)[0], list_test_labels)
        print(f"Epoch {i_epoch+1}", evals_train, evals_test)
        df_evals = df_evals.append({"train-acc": evals_train["accuracy"], "train-f1": evals_train["F1-macro"],
                         "test-acc": evals_test["accuracy"], "test-f1": evals_test["F1-macro"]}, ignore_index=True)
    print(df_evals)
    log_file_name = f"{args.alpha}-{args.batch_size}-{args.dataset}-{args.lr}-{args.n_classes}-{args.n_epoches}-{args.n_features}"
    best_test_f1 = df_evals["test-f1"].max()
    # df_evals.to_csv("log_file/"+log_file_name+".csv")

    df_adjustment = pd.DataFrame(columns=["alpha", "batch_size", "dataset", "lr", "n_classes", "n_epoches", "n_features", "test-acc", "test-f1"])
    best_test_index = df_evals[df_evals["test-f1"]==best_test_f1].index[0]
    with open("log_file/arg_adjustment.txt", "a") as f:
        f.write(log_file_name.replace("-", " ") + f" {args.n_bigram}" + " {} {}\n".format(df_evals.loc[best_test_index]["test-acc"], df_evals.loc[best_test_index]["test-f1"]))