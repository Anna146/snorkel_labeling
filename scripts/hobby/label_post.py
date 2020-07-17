import string
import sys

from collections import defaultdict
from snorkel.labeling import LabelModel, PandasLFApplier, filter_unlabeled_dataframe, labeling_function, LFAnalysis, LabelingFunction
from snorkel.labeling.apply.dask import PandasParallelLFApplier
from nltk import tokenize
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import multiprocessing
from utils import *
import swifter
import pandas as pd
ABSTAIN = -1
NEG = 0
POS = 1

num_cpu = multiprocessing.cpu_count()
hobby_subred_dict = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/hobby_subreddits.txt").read())
hobby_to_syn = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/hobby_synonyms.txt").read())
syn_to_hob = dict((syn, val) for val, syns in hobby_to_syn.items() for syn in syns)

liking = ["obsessed with", "fond of", "keen on", 'like', 'enjoy', 'love', 'play', 'take joy in', 'adore', 'appreciate',
          "fan of", 'fascinated by', "interested in", 'fancy', 'mad about', "practise", "into", "sucker for"]
hating = ["hate", "dislike", "detest", "can't stand"]

# to capture patterns "i enjoy swimming"
def pattern_match(x, keyword, label, neg_label, context_len = 3, with_period = False):
    if x['author'] == 'foo':
        return ABSTAIN
    for txt in x.texts:
        txt_punct = txt
        txt = txt.translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower()
        txt = " " + txt + " "
        txt_punct = " " + txt_punct + " "
        pos_target = txt.find(" " + x.value)
        pos_target_punct = txt_punct.find(" " + x.value)
        if pos_target == -1:
            continue
        if with_period:
            if all(z not in txt_punct[pos_target_punct + len(x.value):pos_target_punct + len(x.value)+ len(z) + 2] for z in [y for y in "!(),./:;[]"] + [" and", " but", " however", " though", " still", " anyway", " -"]):
                continue
        left_cont = " " + " ".join(txt[:pos_target].strip().split(' ')[min(0, - context_len - len(keyword.split(" "))):]) + " "
        if any(y in left_cont for y in [" my ", " your ", " his ", " her ", " its ", " our ", " their ", " theirs ", " the ", " a ", " an ", " this ", " these ", " that ", " those "]):
            continue
        pos_kw = left_cont.find(" " + keyword + " ")
        if pos_kw == -1:
            continue
        pos_i = min(y for y in [left_cont.find(" i "), left_cont.find(" i'm "), sys.maxsize] if y != -1)
        if pos_i == sys.maxsize:
            continue
        if any(z in txt[pos_i:] for z in [y for y in',.?!;:']):
            continue
        pos_not = min(y for y in [left_cont.find(" not "), left_cont.find("don't"), left_cont.find(" never "), sys.maxsize] if y != -1)
        if pos_i < pos_not and pos_not < pos_kw:
            return neg_label
        if pos_i < pos_kw:
            return label
    return ABSTAIN

def make_keyword_lf(keyword, label, neg_label, context_len, with_period):
    return LabelingFunction(
        name="pattern_%s_%s%s" % (keyword, "context:%d" % context_len, "_period" if with_period else ""),
        f=pattern_match,
        resources=dict(keyword=keyword, label=label, neg_label=neg_label, with_period=with_period, context_len=context_len),
    )

#######################################
#####  everyting above with period/comma after, to avoid cases "i don't like that the swimming companion bug still happens"
#######################################


# to match high-confidence "my hobby is swimming", "swimming is my hobby"
@labeling_function()
def very_confident_matcher(x):
    left = ["my hobby ", "my interest "]
    right = [" my interest", " my hobby", " one of my interests", " one of my hobbies", " among my interests", " among my hobbies", " my passion", " my obsession"]
    left_combi = ["i have ", "i do ", "i play "]
    txt = " " + x.text + " "
    for pat in left:
        if txt.find(pat + "is " + x.value) != -1:
            return POS
        if txt.find(pat + "is not " + x.value) != -1:
            return NEG
        if txt.find(pat + "isn't " + x.value) != -1:
            return NEG
    for pat in right:
        if txt.find(x.value + " is" + pat) != -1:
            return POS
        if txt.find(x.value + " is not" + pat) != -1:
            return NEG
        if txt.find(x.value + " isn't" + pat) != -1:
            return NEG
    for pat in left_combi:
        if txt.find(pat + x.value + " as a hobby") != -1:
            return POS
        if txt.find(pat + x.value + " for a hobby") != -1:
            return POS
    return ABSTAIN

# to match high-confidence but loosely: "my only hobby is quick swimming", "swimming is among my interests"
@labeling_function()
def loose_confident_matcher(x):
    for txt in x.texts:
        txt = txt.translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower()
        txt = " " + txt + " "
        pos_target = txt.find(" " + x.value + " ")
        if pos_target == -1:
            continue
        context = " " + " ".join(txt[:pos_target].strip().split(' ')[-5:]) + " " + " ".join(txt[pos_target + len(x.value) + 1:].strip().split(' ')[:5]) + " "
        hobby_here = min(y for y in [context.find(" hobby "), context.find(" hobbies "), context.find(" interest "), context.find(" interests "), sys.maxsize] if y != -1)
        if hobby_here == sys.maxsize:
            continue
        im_here = min(y for y in [context.find(" i "), context.find(" my "), context.find(" i'm "), sys.maxsize] if y != -1)
        if im_here == sys.maxsize:
            continue
        not_here = min(
            y for y in [context.find(" no "), context.find(" not "), context.find(" don't "), context.find(" never "), sys.maxsize] if y != -1)
        if not_here != sys.maxsize:
            return NEG
        return POS
    return ABSTAIN

# to capture frequent activities "I usually go swimming" within sentence
@labeling_function()
def usual_capture(x):
    for txt in x.texts:
        txt = txt.translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower()
        txt = " " + txt + " "
        if x.value not in txt:
            continue
        if all(y not in txt for y in [" usually ", " normally ", " typically ", " generally ", " always "]):
            continue
        pos_i = min([txt.find(" i "), txt.find(" i'm "), txt.find(" my "), sys.maxsize])
        if pos_i == sys.maxsize:
            continue
        not_here = min(y for y in [txt.find(" no "), txt.find(" not "), txt.find(" don't "), txt.find(" never "), sys.maxsize] if y != -1)
        if not_here != sys.maxsize:
            return NEG
        return POS
    return ABSTAIN

# the word hobby is somewhere in the sentence "for hobby i do swimming"
@labeling_function()
def hobby_near(x):
    return POS if "hobby" in x.text.lower() else ABSTAIN

# the word interest(-ed, -ing) is somewhere in the sentence "swimming is my main interest"
@labeling_function()
def interest_near(x):
    return POS if "interest" in x.text.lower() else ABSTAIN

# if the name of the hobby is in subreddit the post is from "mad_swimming"
@labeling_function()
def val_in_subr(x):
    return POS if x.value.lower() in x.subr.lower() else ABSTAIN

# the sentiment between "I" and hobby name is positive "i think the best thing is swimming"
@labeling_function()
def polarity_span(x):
    txt = x.text.lower()
    pos_target = txt.find(" " + x.value + " ")
    pos_i = min([txt.find(" i "), txt.find(" i'm "), txt.find(" my "), sys.maxsize])
    if pos_i == sys.maxsize or pos_i > pos_target:
        return ABSTAIN
    scores = TextBlob(x.text[pos_i:pos_target + len(x.value) + 1]).sentiment
    if scores.polarity > 0.4:
        return POS
    if scores.polarity < -0.4:
        return NEG
    return ABSTAIN

# the overall sentiment of the sentence is positive
@labeling_function()
def polarity_whole_sentence(x):
    for txt in x.texts:
        if x.value not in txt:
            continue
        scores = TextBlob(txt).sentiment
        if scores.polarity > 0.4:
            return POS
        if scores.polarity < -0.4:
            return NEG
    return ABSTAIN

# the overall sentiment of the post is positive
@labeling_function()
def polarity_whole_post(x):
    scores = TextBlob(x.text).sentiment
    if scores.polarity > 0.4:
        return POS
    if scores.polarity < -0.4:
        return NEG
    return ABSTAIN

# posted in hobby-related subreddit
@labeling_function()
def check_subreddit(x):
    if x['author'] == 'foo':
        return ABSTAIN
    if x.subr.lower() in hobby_subred_dict[x.root_value]:
        return POS
    return ABSTAIN

#########################################################################################

################# MAIN ###################################################################

import time
from tqdm_dask_progressbar import TQDMDaskProgressBar

def find_val(hays, need):
    return " ".join([h.find(" " + need) != -1 for h in hays])

def label_post(inp_path, prefix = ""):
    #lfs = [very_confident_matcher, loose_confident_matcher, usual_capture, hobby_near, interest_near, val_in_subr, polarity_span, polarity_whole_sentence, polarity_whole_post, check_subreddit]
    lfs = [very_confident_matcher, loose_confident_matcher, usual_capture, hobby_near, interest_near, polarity_span]

    context_lens = [100, 3]

    for with_per in [True, False]:
        for clen in context_lens:
            for kw in liking:
                lfs.append(make_keyword_lf(keyword=kw, label=POS, neg_label=NEG, context_len=clen, with_period=with_per))
            for kw in hating:
                lfs.append(make_keyword_lf(keyword=kw, label=NEG, neg_label=POS, context_len=clen, with_period=with_per))

    print("created lfs, their count", len(lfs))

    t1 = time.time()

    df_train = pd.read_pickle(inp_path)

    df_train['texts'] = df_train['text'].swifter.apply(lambda x: [y.lower() for y in tokenize.sent_tokenize(x)])
    df_train['root_value'] = df_train['value'].swifter.set_dask_threshold(dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: syn_to_hob[x])
    #df_train['containing_sentences'] = df_train[['texts', 'value']].swifter.apply(lambda y: find_val(y['texts'], y['value']), axis=1)

    print("loaded dataset")

    t1 = time.time()
    with TQDMDaskProgressBar(desc="Dask Apply"):
        applier = PandasParallelLFApplier(lfs=lfs)
        L_train = applier.apply(df=df_train, n_parallel=num_cpu)
    print("time mins ", (time.time() - t1) / 60)

    verbose = True
    if verbose:
        for i in range(len(lfs)):
            ppath = "/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/interesting_datasets/" + str(lfs[i]).split(",")[0] + ".csv"
            df_train.iloc[L_train[:, i] != ABSTAIN].to_csv(ppath)

    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    df_l_train = pd.DataFrame(L_train, columns=[str(x).split(",")[0] for x in lfs])
    print(df_train.shape)
    print(df_l_train.shape)
    df_train = pd.concat([df_train.reset_index(), df_l_train], axis=1)
    print(df_train.shape)
    print("*******************************************")
    df_train = df_train.drop(["index"], axis=1)
    df_train.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/tmp_train_post_" + prefix + ".csv")

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=1000, lr=0.001, log_freq=100, seed=123)
    probs_train = label_model.predict_proba(L=L_train)

    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=probs_train, L=L_train
    )

    df_train_filtered.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/fi_train_post_" + prefix + ".csv")

    print("the length of unfiltered posts", len(set(df_train['author'] + "+++++" + df_train['value'])))
    print("the length of filtered posts", len(set(df_train_filtered['author'] + "+++++" + df_train_filtered['value'])))

    probs_df = pd.DataFrame(probs_train_filtered, columns=["neg_prob", "pos_prob"])
    print(df_train_filtered.shape)
    print(probs_df.shape)
    df_train_filtered = pd.concat([df_train_filtered.reset_index(), probs_df], axis=1)
    print(df_train_filtered.shape)
    df_train_filtered.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/train_post_" + prefix + ".pkl")
    df_train_filtered.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/train_post_" + prefix + ".csv")

    #df_train.iloc[L_train[:, 1] != ABSTAIN].to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/intr_train_post_tmp.csv")

    auth_hobby_dict = defaultdict(set)
    for index, row in df_train.iterrows():
        if row.value == row.value and row.author == row.author:
            auth_hobby_dict[row.author].add(row.value)

    with open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/author_hobby_dict_" + prefix + ".txt", "w") as f_out:
        f_out.write(repr(dict(auth_hobby_dict)))