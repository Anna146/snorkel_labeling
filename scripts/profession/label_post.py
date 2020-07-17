import string
import sys

from collections import defaultdict
from snorkel.labeling import LabelModel, PandasLFApplier, filter_unlabeled_dataframe, labeling_function, LFAnalysis, LabelingFunction
from snorkel.labeling.apply.dask import PandasParallelLFApplier
from nltk import tokenize
import multiprocessing
import swifter
import pandas as pd
ABSTAIN = -1
NEG = 0
POS = 1

num_cpu = multiprocessing.cpu_count()
hobby_subred_dict = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_subreddits.txt").read())
hobby_to_syn = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_synonyms.txt").read())
syn_to_hob = dict((syn, val) for val, syns in hobby_to_syn.items() for syn in syns)
syn_list = list(syn_to_hob.keys())

patterns = ["i am a", "i am an", "i'm an", "i'm a", "my profession", "i work as", "my job", "my occupation", "i regret becoming"]

# to capture patterns "i am a student"
# with_period - to remove false matches in the right context: "i am a student teacher"
def pattern_match(x, keyword, context_len = 3, with_period = False):
    if x['author'] == 'foo':
        return ABSTAIN
    for txt in x.texts:
        txt_punct = txt
        txt = txt.translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower()
        txt = " " + txt + " "
        txt_punct = " " + txt_punct.strip() + ". "
        pos_target = txt.find(" " + x.value)
        pos_target_punct = txt_punct.find(" " + x.value)
        if pos_target == -1:
            continue
        if with_period:
            if all(z not in txt_punct[pos_target_punct + len(x.value):pos_target_punct + len(x.value)+ len(z) + 2] for z in [y for y in "!(),./:;[]"] + [" and", " but", " however", " though", " still", " anyway", " -"]):
                continue # btw there is a problem with comma, I remove it because I write in csv format on hadoop
        left_cont = " " + " ".join(txt[:pos_target].strip().split(' ')[min(0, - context_len - len(keyword.split(" "))):]) + " "
        # clean left context: if user says other_prof and prof, move the actual value closer to the pattern
        split_left_cont = left_cont.strip().split(" ")
        if len(split_left_cont) > 2:
            if split_left_cont[-1] == "and" and split_left_cont[-2] in syn_list:
                left_cont = " " + " ".join(split_left_cont[:-2]) + " "
        ####
        pos_kw = left_cont.find(" " + keyword + " ")
        if pos_kw == -1:
            continue
        if any(z in txt[pos_kw:] for z in [y for y in',.?!;:']):
            continue
        pos_not = min(y for y in [left_cont.find(" not "), left_cont.find("don't"), left_cont.find(" no "), sys.maxsize] if y != -1)
        if pos_kw < pos_not and pos_not != sys.maxsize:
            # eliminate the cases "I am not THE doctor", "I am not HIS doctor", where despite negation, it is an actual profession
            if any(y in left_cont[pos_not:] for y in [" your ", " his ", " her ", " their ", " the ", " this ", " that "]) and keyword in ["i am a", "i am an", "i'm an", "i'm a"]:
                return POS
            return NEG
        return POS
    return ABSTAIN

def make_keyword_lf(keyword, context_len, with_period):
    return LabelingFunction(
        name="pattern_%s_%s%s" % (keyword, "context:%d" % context_len, "_period" if with_period else ""),
        f=pattern_match,
        resources=dict(keyword=keyword, with_period=with_period, context_len=context_len),
    )

###############

# synonyms of "profession" are in the post: "being a student is my career"
@labeling_function()
def job_inpost(x):
    job_synonyms = ["profession", "job", "career", "occupation", "duty", "position", "speciality", "employment"]
    return POS if any([y in x.text.lower() for y in job_synonyms]) else ABSTAIN

# posted in profession-related subreddit
@labeling_function()
def check_subreddit(x):
    if x['author'] == 'foo':
        return ABSTAIN
    syns = hobby_to_syn[x.root_value]
    if x.subr.lower() in hobby_subred_dict[x.root_value] or any(["".join(y.split(" ")) in x.subr.lower() for y in syns]) or any(["".join(y.split("_")) in x.subr.lower() for y in syns]):
        return POS
    return ABSTAIN

# posted in iama
@labeling_function()
def check_iama(x):
    if x['author'] == 'foo':
        return ABSTAIN
    if x.subr.lower() in ["iama", "ama"]:
        for txt in x.texts:
            txt = txt.translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower()
            txt = " " + txt + " "
            pos_target = txt.find(" " + x.value)
            if pos_target == -1:
                continue
            if any(y in txt[:pos_target] for y in ["i am a", "i am an", "i'm an", "i'm a"]):
                return POS
    return ABSTAIN

#########################################################################################

################# MAIN ###################################################################

import time
from tqdm_dask_progressbar import TQDMDaskProgressBar

def find_val(hays, need):
    return " ".join([h.find(" " + need) != -1 for h in hays])

def label_post(inp_path, prefix = ""):

    #lfs = [job_inpost, check_subreddit, check_iama]
    lfs = [job_inpost, check_iama]

    context_lens = [100, 3, 2]
    for with_per in [True, False]:
        for clen in context_lens:
            for kw in patterns:
                lfs.append(make_keyword_lf(keyword=kw, context_len=clen, with_period=with_per))

    print("created lfs, their count", len(lfs))

    df_train = pd.read_pickle(inp_path)

    df_train['texts'] = df_train['text'].swifter.apply(lambda x: [y.lower() for y in tokenize.sent_tokenize(x)])
    df_train['root_value'] = df_train['value'].swifter.apply(lambda x: syn_to_hob[x])
    #df_train['containing_sentences'] = df_train[['texts', 'value']].swifter.apply(lambda y: find_val(y['texts'], y['value']), axis=1)

    print("loaded dataset")

    t1 = time.time()
    with TQDMDaskProgressBar(desc="Dask Apply"):
        applier = PandasParallelLFApplier(lfs=lfs)
        L_train = applier.apply(df=df_train, n_parallel=num_cpu)
    print("time mins ", (time.time() - t1) / 60)

    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    df_l_train = pd.DataFrame(L_train, columns=[str(x).split(",")[0] for x in lfs])
    print(df_train.shape)
    print(df_l_train.shape)
    df_train = pd.concat([df_train.reset_index(), df_l_train], axis=1)
    print(df_train.shape)
    print("*************************************************")
    df_train = df_train.drop(["index"], axis=1)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=1000, lr=0.001, log_freq=100, seed=123)
    probs_train = label_model.predict_proba(L=L_train)

    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=probs_train, L=L_train
    )
    print("the length of unfiltered posts", len(set(df_train['author'] + "+++++" + df_train['value'])))
    print("the length of filtered posts", len(set(df_train_filtered['author'] + "+++++" + df_train_filtered['value'])))

    probs_df = pd.DataFrame(probs_train_filtered, columns=["neg_prob", "pos_prob"])
    print(df_train_filtered.shape)
    print(probs_df.shape)
    df_train_filtered = pd.concat([df_train_filtered.reset_index(), probs_df], axis=1)
    print(df_train_filtered.shape)

    df_train_filtered.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/train_post_" + prefix + ".pkl")
    df_train_filtered.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/train_post_" + prefix + ".csv")

    #df_train.iloc[L_train[:, 1] != ABSTAIN].to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/intr_train_post_tmp.csv")

    verbose = True
    if verbose:
        for i in range(len(lfs)):
            ppath = "/home/tigunova/PycharmProjects/snorkel_labels/data/profession/interesting_datasets/" + str(lfs[i]).split(",")[0] + ".csv"
            df_train.iloc[L_train[:, i] != ABSTAIN].to_csv(ppath)


    auth_hobby_dict = defaultdict(set)
    for index, row in df_train.iterrows():
        if row.value == row.value and row.author == row.author:
            auth_hobby_dict[row.author].add(row.value)

    with open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/author_profession_dict_" + prefix + ".txt", "w") as f_out:
        f_out.write(repr(dict(auth_hobby_dict)))