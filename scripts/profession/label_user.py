import csv
import multiprocessing
from collections import defaultdict, Counter
import time

import numpy as np
from snorkel.labeling import LabelModel, PandasLFApplier, filter_unlabeled_dataframe, labeling_function, LFAnalysis, \
    LabelingFunction
from snorkel.labeling.apply.dask import PandasParallelLFApplier
from snorkel.labeling import LabelModel
from profession_utils import *
import pandas as pd

import sys

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 200)

num_cpu = multiprocessing.cpu_count()

ABSTAIN = -1
NEG = 0
POS = 1

epath_cats = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/empath_dict.txt").read())
hobby_to_syn = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_synonyms.txt").read())
syn_to_hob = dict((syn, val) for val, syns in hobby_to_syn.items() for syn in syns)


# count of lf matches for posts is greater than threshold or
# number of empath term matches over threshold or
# number of related subreddits over threshold or
# number of overall good posts
def more_than_treshold(x, thresh, col_name, next_threshold = sys.maxsize):
    if x.author == "foo":
        return ABSTAIN
    val_to_check = x[col_name]
    if val_to_check > thresh and "neg" in col_name.lower() and val_to_check <= next_threshold:
        return NEG
    if val_to_check > thresh and val_to_check <= next_threshold:
        return POS
    return ABSTAIN


def make_thresold_lf(thresh, col_name, next_threshold = sys.maxsize):
    return LabelingFunction(
        name="more_than_%s_%s" % (thresh, col_name),
        f=more_than_treshold,
        resources=dict(thresh=thresh, col_name=col_name, next_threshold = next_threshold),
    )


# profession name is in username "^^cool_programmer^^"
@labeling_function()
def val_in_name(x):
    return POS if x.value.lower() in x.author.lower() else ABSTAIN


# user has to have min num of related lexicon
def less_than_treshold(x, thresh, previous_threshold = -sys.maxsize):
    if x.author == "foo":
        return ABSTAIN
    if x.lexicon_counts < thresh and x.lexicon_counts >= previous_threshold:
        return NEG
    return ABSTAIN

def make_lexicon_lf(thresh, pref = "", previous_threshold = -sys.maxsize):
    return LabelingFunction(
        name="%s_less_%s" % (pref, thresh),
        f=less_than_treshold,
        resources=dict(thresh=thresh, previous_threshold = previous_threshold),
    )

######### workers

def worker_lf(x, worker_dict):
    if x.author == "foo":
        return ABSTAIN
    return worker_dict.get(x.author, ABSTAIN)

def make_annotator_lf(worker_index):
    reader = csv.reader(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/labeling_lf/%d.csv" % (worker_index)))
    next(reader)
    worker_dict = dict((x[-1], POS if x[1] == "checked" else ABSTAIN) for x in reader)
    return LabelingFunction(
        name="worker_%d" % (worker_index),
        f=worker_lf,
        resources=dict(worker_dict=worker_dict),
    )

################

# lexicon function
@labeling_function()
def positive_lexicon_overall(x):
    if x.author == "foo":
        return ABSTAIN
    return POS if x.lexicon_counts > thresh_by_value[x.root_value] else NEG

@labeling_function()
def positive_lexicon_pervalue(x):
    if x.author == "foo":
        return ABSTAIN
    return POS if x.lexicon_counts > overall_thresh else NEG

################# MAIN ###################################################################

from sklearn.metrics import precision_score
from tqdm_dask_progressbar import TQDMDaskProgressBar
import sys
np.set_printoptions(threshold=sys.maxsize)

def label_user(inp_path, prefix=""):
    df_train = pd.read_pickle(inp_path)

    ########## threshold on word similarity
    take_first = 100
    overall_first = 10000
    global thresh_by_value, overall_thresh
    df_train['root_value'] = df_train['value'].swifter.set_dask_threshold(dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: syn_to_hob[x])
    thresh_by_value = df_train.groupby(["root_value"]).apply(lambda x: np.partition(x['lexicon_counts'], max(len(x['lexicon_counts']) - take_first, 0))[max(len(x['lexicon_counts']) - take_first, 0)]).to_dict()
    overall_thresh = np.partition(df_train["lexicon_counts"].to_numpy(), max(len(df_train) - overall_first, 0))[max(len(df_train) - overall_first, 0)]
    print(overall_thresh)
    #############################

    # separately loose - strict, pos - neg, period - without
    names_pool = ["context:2_count_pos", "context:3_count_pos", "context:100_count_pos", "context:2_period_count_pos",
                  "context:3_period_count_pos", "context:100_period_count_pos",
                  "context:2_count_neg", "context:3_count_neg", "context:100_count_neg", "context:2_period_count_neg",
                  "context:3_period_count_neg", "context:100_period_count_neg"]
    for f_name in names_pool:
        curr_cols = [x for x in df_train.columns if f_name in x]
        df_train['total_' + f_name] = df_train[curr_cols].swifter.apply(sum, axis=1)
        df_train = df_train.drop(curr_cols, axis=1)
    for p in ["pos", "neg"]:
        df_train["new_total_context:100_count_" + p] = df_train[["total_context:100_count_" + p, "total_context:3_count_" + p]].swifter.apply(lambda x: max(0, x["total_context:100_count_" + p] - x["total_context:3_count_" + p]), axis=1)
        df_train["new_total_context:3_count_" + p] = df_train[["total_context:3_count_" + p, "total_context:2_count_" + p]].swifter.apply(lambda x: max(0, x["total_context:3_count_" + p] - x["total_context:2_count_" + p]), axis=1)
        df_train["new_total_context:100_period_count_" + p] = df_train[["total_context:3_period_count_" + p, "total_context:100_period_count_" + p]].swifter.apply(lambda x: max(0, x["total_context:100_period_count_" + p] - x["total_context:3_period_count_" + p]), axis=1)
        df_train["new_total_context:3_period_count_" + p] = df_train[["total_context:3_period_count_" + p, "total_context:2_period_count_" + p]].swifter.apply(lambda x: max(0, x["total_context:3_period_count_" + p] - x["total_context:2_period_count_" + p]), axis=1)
        df_train["new_total_context:2_count_" + p] = df_train[["total_context:100_period_count_" + p, "total_context:2_count_" + p]].swifter.apply(lambda x: max(0, x["total_context:2_count_" + p] - x["total_context:100_period_count_" + p]), axis=1)

    df_train = df_train.drop(["total_" + x for x in names_pool if "2_period_count" not in x], axis=1)

    lfs = [val_in_name, positive_lexicon_overall, positive_lexicon_pervalue]
    num_of_thesholds = 3
    step = 100 // num_of_thesholds

    for col in df_train:
        if col not in ["author", "value", "idd", "root_value"]:
            if col not in ["pos_prob_mean", "neg_prob_mean",
                           "num_good_posts"]:  # , "lexicon_counts", "subreddit_counts", "name_in_subr_count"]:
                thresholds = [0]
                if "lexicon" in col and "unique" not in col:
                    continue
                if True:  # col in ["lexicon_counts", "unique_lexicon_counts"]:
                    vals = df_train[col].to_numpy()
                    thresholds = np.percentile(vals, list(range(0 + step, 99 + step, step))).astype(int)
                    thresholds = sorted(list(set(thresholds)))
                    if len(thresholds) > 1:
                        thresholds = thresholds[:-1]
                    if "lexicon" in col:
                        thresholds = [3]
                    # max_val = max(vals)
                    # thresholds = list(range(0, int(max_val), int(max_val/5) + 1))
                # elif col == "pos_prob_mean":
                #    thresholds = [0.5 + 0.1 * x for x in range(5)]
                for i in range(len(thresholds)):
                    thresh = thresholds[i]
                    next_threshold = sys.maxsize if i == len(thresholds) - 1 else thresholds[i+1]
                    previous_threshold = -sys.maxsize if i == 0 else thresholds[i-1]
                    if "lexicon_counts" not in col:
                        lfs.append(make_thresold_lf(thresh=thresh, col_name=col, next_threshold=next_threshold))
                    else:
                        lfs.append(make_lexicon_lf(thresh=thresh, pref=col, previous_threshold = previous_threshold))

    num_annotators = 0
    if num_annotators > 0:
        for i in range(1, num_annotators + 1):
            lfs.append(make_annotator_lf(worker_index=i))

    lfs = [x for x in lfs if any(y in str(x) for y in ["less", "context:2", "worker", "lexicon"])]
    print("created lfs their number", len(lfs))
    print("\n".join(str(x) for x in lfs))

    #### validation #####
    do_val = False
    if do_val:
        df_golden = pd.read_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/gold_dev.csv")
        name_val = list(df_golden["auth_val"])
        # df_train['root_value'] = df_train['value'].swifter.apply(lambda x: syn_to_hob[x])
        df_train["auth_val"] = df_train[["author", "value"]].swifter.apply(lambda x: x["author"] + "+++" + x["value"],
                                                                           axis=1)
        df_val = df_train[df_train.auth_val.isin(name_val)]
        df_dev = df_train[~df_train.auth_val.isin(name_val)]
        print("Number val", df_val.shape)
        print("Number dev", df_dev.shape)
        df_val = df_val.merge(df_golden, on="auth_val")
        y_val = np.array(df_val["final"])
        df_val = df_val.drop(labels="final", axis=1)
        # create test set as well

        with TQDMDaskProgressBar(desc="Dask Apply"):
            applier = PandasParallelLFApplier(lfs=lfs)
            L_val = applier.apply(df=df_val, n_parallel=num_cpu)
            L_dev = applier.apply(df=df_dev, n_parallel=num_cpu)

        dev_analysis = LFAnalysis(L=L_dev, lfs=lfs).lf_summary()
        analysis = LFAnalysis(L=L_val, lfs=lfs).lf_summary(y_val)
        analysis.to_csv("/home/tigunova/val_analysis.csv")
        dev_analysis.to_csv("/home/tigunova/dev_analysis.csv")
        print(analysis)
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_dev)#, Y_dev=y_val)
        model_stat = label_model.score(L=L_val, Y=y_val)
        print(model_stat)
        exit(0)
    ###########

    #### picking threshold #####
    do_threshold = False
    if do_threshold:
        df_golden = pd.read_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/gold_validation.csv")
        name_val = list(df_golden["auth_val"])
        # df_train['root_value'] = df_train['value'].swifter.apply(lambda x: syn_to_hob[x])
        df_train["auth_val"] = df_train[["author", "value"]].swifter.apply(
            lambda x: x["author"] + "+++" + x["value"], axis=1)
        df_val = df_train[df_train.auth_val.isin(name_val)]
        df_dev = df_train[~df_train.auth_val.isin(name_val)]
        pop_size = df_dev.shape[0]
        print("Number val", df_val.shape)
        print("Number dev", df_dev.shape)
        applier = PandasParallelLFApplier(lfs=lfs)
        df_val = df_val.merge(df_golden, on="auth_val")
        L_val = applier.apply(df=df_val, n_parallel=num_cpu)
        val_thresholds = [0.01 * x for x in range(100)]
        label_model = LabelModel(cardinality=2, verbose=True)
        with TQDMDaskProgressBar(desc="Dask Apply"):
            L_dev = applier.apply(df=df_dev, n_parallel=num_cpu)
            label_model.fit(L_dev, class_balance=[0.5, 0.5])  # , Y_dev=y_val)
            wghts = label_model.get_weights()
            print("\n".join(str(x) for x in zip(lfs, wghts)))
            probs_val = label_model.predict_proba(L=L_val)
            probs_df = pd.DataFrame(probs_val, columns=["neg_prob", "pos_prob"])
            df_val = pd.concat([df_val.reset_index(), probs_df], axis=1)
            probs_dev = label_model.predict_proba(L=L_dev)
            probs_df = pd.DataFrame(probs_dev, columns=["neg_prob", "pos_prob"])
            df_dev = pd.concat([df_dev.reset_index(), probs_df], axis=1)
            y_true = np.array(df_val["final"])
        for th in val_thresholds:
            y_pred = np.array(df_val["pos_prob"].apply(lambda x: 1 if x > th else 0))
            #print("true negatives")
            #print(df_val[df_val["final"] == 1][df_val["pos_prob"] <= th][["auth_val", "text"]])
            prec = precision_score(y_true, y_pred)

            pred_labels = y_pred
            true_labels = y_true
            # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
            TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

            print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))

            # print(list(zip(label_model.predict(L=L_val_curr), y_val_curr)))
            # print("******************************")
            print("threshold %s, proportion population %.4f, precision %s" % (
            str(th), df_dev[df_dev["pos_prob"] > th].shape[0] / pop_size, str(prec)))
        exit(0)
    ###########

    with TQDMDaskProgressBar(desc="Dask Apply"):
        applier = PandasParallelLFApplier(lfs=lfs)
        L_train = applier.apply(df=df_train, n_parallel=num_cpu)

    analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print(analysis)

    df_l_train = pd.DataFrame(L_train, columns=["llf_" + str(x).split(",")[0] for x in lfs])
    print(df_train.shape)
    print(df_l_train.shape)
    df_train = pd.concat([df_train.reset_index(), df_l_train], axis=1)
    print(df_train.shape)
    print("********************************************")

    t4 = time.time()
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=1000, lr=0.001, log_freq=100, seed=123, class_balance=[0.3, 0.7])

    probs_train = label_model.predict_proba(L=L_train)
    print("labeling model work ", (time.time() - t4) / 60)

    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=probs_train, L=L_train
    )

    probs_df = pd.DataFrame(probs_train_filtered, columns=["neg_prob", "pos_prob"])
    print(df_train_filtered.shape)
    print(probs_df.shape)
    result_filtered = pd.concat([df_train_filtered[['author', 'value', 'idd']].reset_index(), probs_df], axis=1)
    print(result_filtered.shape)
    print("****************************************************")

    result_filtered.to_csv("/home/tigunova/some_result_" + prefix + ".csv")

    print(df_train_filtered.shape)
    print(probs_df.shape)
    df_train_filtered = pd.concat([df_train_filtered.reset_index(), probs_df], axis=1)
    df_train_filtered = df_train_filtered.drop(["index"], axis=1)
    print(df_train_filtered.shape)
    df_train_filtered.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/user_" + prefix + ".pkl")
    df_train_filtered.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/user_" + prefix + ".csv")

    # df_train.iloc[L_train[:, 1] == POS].to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/user_" + prefix + ".csv")

    ### write dict
    output_threshold = 0.63
    output_dict = defaultdict(list)
    auth_hobby_dict = defaultdict(list)
    for index, row in result_filtered.iterrows():
        if row.value == row.value and row.author == row.author:
            auth_hobby_dict[row.author].append([row.value, row.pos_prob])

    allowed_labels = []
    for index, row in df_train_filtered.iterrows():
        if row.value == row.value and row.author == row.author:
            if row.pos_prob > output_threshold:
                output_dict[row.author].append([row.value] + row.idd + [row.pos_prob])
                allowed_labels.append(syn_to_hob[row.value])
    print("\n".join([str(y) for y in sorted(dict(Counter(allowed_labels)).items(), key=lambda x: x[1])]))
    print("After cropping", sum([x if x < 500 else 500 for x in dict(Counter(allowed_labels)).values()]))
    print("users in total", len(output_dict))
    for auth, stuffs in output_dict.items():
        prof = ":::".join(set([x[0] for x in stuffs]))
        prob = ":::".join([str(x[-1]) for x in stuffs])
        msgs = set([x for l in stuffs for x in l[1:-1]])
        output_dict[auth] = [prof] + list(msgs) + [prob]

    with open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/final_author_dict_" + prefix + ".txt",
              "w") as f_out:
        f_out.write(repr(dict(auth_hobby_dict)))
    with open("/home/tigunova/users_profession1.txt", "w") as f_out:
        f_out.write(repr(dict(output_dict)))



