import os

import numpy as np
import pandas as pd

from nltk import tokenize

ABSTAIN = -1
NEG = 0
POS = 1

def load_keywords():
    break_trigrams = False
    folder = "/home/tigunova/yake_keywords/bigrams"
    kwrds = []
    thresh = 1000
    for fi in os.listdir(folder):
        with open(os.path.join(folder, fi)) as f_in:
            lines = f_in.readlines()
            if not break_trigrams:
                kwrds.extend([x.strip().split(",")[0] for x in lines[-thresh:]])
            else:
                nspl = [x.strip().split(",")[0] for x in lines[-thresh:]]
                spl = []
                for x in nspl:
                    spl.append(x)
                    x = x.split(" ")
                    if len(x) == 3:
                        spl.append(" ".join([x[0],x[1]]))
                        spl.append(" ".join([x[1],x[2]]))
                        spl.append(" ".join([x[0],x[2]]))
                kwrds.extend(spl)
    return kwrds

def process_forum_dataset():
    data_path = "/GW/D5data-11/achetan/tripadvisor_scraping/forum_dumps/old/1.json"
    kwrds = load_keywords()
    frame = pd.read_json(data_path, lines=True)
    train_set = pd.DataFrame(columns=["target", "text", "author", "turn", "header"])
    i = 0
    for conv in frame['conversation']:
        for turn in conv:
            for sent in tokenize.sent_tokenize(turn['content'].replace("\n"," ").replace(".", ". ")):
                for wrd in kwrds:
                    if len(wrd.split(" ")) > 1 and (wrd + " " in sent.lower() or " " + wrd in sent.lower()):
                        train_set.loc[i] = [wrd, sent, turn['author']['uname'], turn['turn'], turn['header']]
                        i += 1
    train_set.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/target_dataframe.pkl")
    train_set.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/target_dataframe.csv")

def process_labeled():
    dev_set = pd.DataFrame(columns=["target", "text", "author", "turn", "header", "label"])
    i = 0
    with open("/home/tigunova/Downloads/target_emotion.txt") as f:
        for line in f:
            if len(line) < 2:
                continue
            line = line.strip().split("\t")
            dev_set.loc[i] = [line[2].split("-")[1].strip(), line[1], "xyz", "1" , "zyx", 2]
            i += 1
    dev_set.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/dev_target_dataframe.pkl")



#####################    FOR REDDIT     ###########################################


lexicon = [line.strip() for line in open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/empath_lex.txt").readlines()]

hobby_to_syn = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/hobby_synonyms.txt").read())
syn_to_hob = dict((syn, val) for val, syns in hobby_to_syn.items() for syn in syns)

dating_subs = [x.strip().lower() for x in open("/home/tigunova/PycharmProjects/snorkel_labels/data/dating_list.txt").readlines()]

similarities_dict = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/similarities_empath.txt").read())

import time

def process_reddit_post_csv(data_path, prefix = ""):
    train_set = pd.read_csv(data_path, header=None, names = ["author", "value", "subr", "idd", "text"])
    train_set.drop(columns=["idd"])
    bad_subs = set([x for x in train_set.subr if any(y in x.lower() for y in ["meet", "penpals"]) or x.lower() in dating_subs])
    train_set = train_set[~train_set.subr.isin(bad_subs)]
    #train_set = train_set[train_set.subr not in ["chat"] and not any([x in train_set.subr for x in ["chat", "meet", "penpals"]])]
    train_set.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/raw_train_posts_" + prefix + ".pkl")
    train_set.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/raw_train_posts_" + prefix + ".csv")

import csv
import swifter
import _pickle as pickle
import base64

def process_reddit_user_raw(data_path, prefix=""):
    t0 = time.time()
    author_hob_dict = eval(
        open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/author_hobby_dict_" + prefix + ".txt").read())
    hobby_empath_dict = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/empath_dict.txt").read())
    #hobby_subred_dict = eval(
    #    open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/hobby_subreddits.txt").read())
    empath_list = dict((y, str(x)) for x,y in enumerate([line.strip() for line in
                   open("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/sources/empath_lex.txt").readlines()]))
    hobby_empath_dict_inverse = dict((k,[empath_list[x.strip()] for x in v]) for k,v in hobby_empath_dict.items())
    train_set = pd.read_csv(data_path, header=None, names=["author", "words", "counts"])
    train_set = train_set.drop(["counts"], axis=1)
    old_len = train_set.shape
    train_set = train_set[train_set.author.isin(author_hob_dict)]
    train_set['words'] = train_set['words'].swifter.apply(lambda x: pickle.loads(base64.b64decode(x.strip(), "-_")))
    #train_set['subreddits'] = train_set['subreddits'].swifter.set_dask_threshold(dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: pickle.loads(base64.b64decode(x.strip(), "-_")))
    train_set['vals'] = train_set['author'].swifter.apply(lambda x: list(author_hob_dict[x]))
    train_set = train_set.vals.swifter.apply(pd.Series) \
        .merge(train_set, right_index=True, left_index=True) \
        .drop(["vals"], axis=1) \
        .melt(id_vars=["author", "words"], value_name="value") \
        .drop("variable", axis=1) \
        .dropna()
        #.melt(id_vars=["author", "words", "subreddits"], value_name="value") \
    train_set['root_value'] = train_set['value'].swifter.set_dask_threshold(dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: syn_to_hob[x])
    t1 = time.time()
    #train_set['words'] = train_set[['words', 'root_value']].swifter.set_dask_threshold(dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: [x['words'][y] for y in hobby_empath_dict_inverse[x['root_value']] if y in x['words']], axis=1)
    #train_set['lexicon_counts'] = train_set['words'].swifter.apply(lambda x: sum(x), axis=1)
    #train_set['unique_lexicon_counts'] = train_set['words'].swifter.apply(lambda x: len(x), axis=1)
    #print("calculated total")
    #train_set['word_sims'] = train_set[['words', 'root_value']].swifter.set_dask_threshold(dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: sum([x['words'][y] * similarities_dict[x['root_value']][y] for y in x['words'] if y in similarities_dict[x['root_value']]]) , axis=1)
    train_set['lexicon_counts'] = train_set[['words', 'root_value']].swifter.set_dask_threshold(dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: sum([x['words'][y] * similarities_dict[x['root_value']][int(y)] for y in x['words'] if int(y) in similarities_dict[x['root_value']]]) / max(sum([x['words'][y] for y in x['words'] if int(y) in similarities_dict[x['root_value']]]), 1), axis=1)
    print("calculated sims")
    #train_set['lexicon_counts'] = train_set[['word_sum', 'word_sims']].swifter.apply(lambda x: x["word_sims"] / max(x["word_sum"],1), axis=1)
    print("finished with words")
    print("time for lex", time.time() - t1)
    train_set = train_set.drop(["words"], axis=1)
    #train_set['subreddit_counts'] = train_set[['subreddits', 'value', 'root_value']].swifter.apply(lambda x: sum(x['subreddits'][y] for y in hobby_subred_dict[x['root_value']] if y in x['subreddits']) +                                                                                         sum(x['subreddits'][y] for y in x['subreddits'] if "_".join(x['value'].split(" ")) or "".join(x['value'].split(" ")) in y or "_".join(x['root_value'].split(" ")) or "".join(x['root_value'].split(" ")) in y), axis=1)
    #train_set['name_in_subr_count'] = train_set[['subreddits', 'value', 'root_value']].swifter.apply(lambda x: sum(x['value'] in y or x['root_value'] in y for y in x['subreddits']), axis=1)
    train_set = train_set.drop(["root_value"], axis=1)
    #train_set = train_set.drop(["subreddits"], axis=1)
    print("initially author-values ", len(author_hob_dict), " num authors in train set ", old_len, "  written records ", train_set.shape)
    print("total time mins", (time.time() - t0) / 60)
    train_set.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/train_authors_" + prefix + ".pkl")
    train_set.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/train_authors_" + prefix + ".csv")

def join_post_author(post_path, author_path, prefix = ""):
    posts = pd.read_pickle(post_path)
    authors = pd.read_pickle(author_path)
    print("the length of posts", len(set(posts['author'] + "+++++" + posts['value'])))
    print("the length of authors", len(set(authors['author'] + "+++++" + authors['value'])))
    resulting = pd.merge(posts, authors, on=['author', 'value'], how='inner')
    print("the length of joint", len(set(resulting['author'] + "+++++" + resulting['value'])))
    resulting.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/joint_post_author_" + prefix + ".pkl")
    resulting.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/joint_post_author_" + prefix + ".csv")

def count_neg(x):
    return int(np.sum(y == 0 for y in x))

def count_pos(x):
    return int(np.sum(y == 1 for y in x))

def to_lst(x):
    return [y for y in x]

import multiprocessing as mp

def f3(df):
    print("function3 is performed by ", os.getpid())
    ddf3 = df[["author", "value", 'idd']].groupby(["author", "value"]).agg([to_lst])
    ddf3 = ddf3.reset_index()
    ddf3.columns = ddf3.columns.get_level_values(0)
    return ddf3

def f22(df):
    print("function22 is performed by ", os.getpid())
    ddf22 = df[["author", "value", 'neg_prob', 'pos_prob']].groupby(["author", "value"]).agg(['count'])
    ddf22 = ddf22.reset_index()
    ddf22.columns = ["_".join(x) if x[1] != "" else x[0] for x in
                    zip(ddf22.columns.get_level_values(0), ddf22.columns.get_level_values(1))]
    ddf22 = ddf22.drop(["neg_prob_count"], axis=1)
    ddf22 = ddf22.rename(columns={"pos_prob_count": "num_good_posts"})
    return ddf22

def f21(df):
    print("function21 is performed by ", os.getpid())
    ddf2 = df[["author", "value", 'neg_prob', 'pos_prob']].groupby(["author", "value"]).agg(['mean'])
    ddf2 = ddf2.reset_index()
    ddf2.columns = ["_".join(x) if x[1] != "" else x[0] for x in
                    zip(ddf2.columns.get_level_values(0), ddf2.columns.get_level_values(1))]
    return ddf2

def f12(df):
    print("function12 is performed by ", os.getpid())
    ddf = df.drop([z for z in df.columns if "LabelingFunction" not in z and z != "author" and z != "value"],
                        axis=1).groupby(["author", "value"]).aggregate([count_neg])
    ddf = ddf.reset_index(col_fill="")
    ddf.columns = ["_".join(x) if x[1] != "" else x[0] for x in
                   zip(ddf.columns.get_level_values(0), ddf.columns.get_level_values(1))]
    return ddf

def f11(df):
    print("function11 is performed by ", os.getpid())
    ddf = df.drop([z for z in df.columns if "LabelingFunction" not in z and z != "author" and z != "value"],
                        axis=1).groupby(["author", "value"]).aggregate([count_pos])
    ddf = ddf.reset_index(col_fill="")
    ddf.columns = ["_".join(x) if x[1] != "" else x[0] for x in
                   zip(ddf.columns.get_level_values(0), ddf.columns.get_level_values(1))]
    return ddf

def merge_post_df(df_train_path, prefix = ""):
    t1 = time.time()
    df_train = pd.read_pickle(df_train_path)

    pool = mp.Pool(processes=mp.cpu_count())
    print("thread num", mp.cpu_count())
    ddf3 = pool.apply_async(f3, (df_train,))
    ddf22 = pool.apply_async(f22, (df_train,))
    ddf21 = pool.apply_async(f21, (df_train,))
    ddf12 = pool.apply_async(f12, (df_train,))
    ddf11 = pool.apply_async(f11, (df_train,))
    pool.close()
    pool.join()

    df_res = pd.concat([ddf12.get().reset_index().drop(['index'], axis=1), ddf11.get().reset_index().drop(["author", "value", 'index'], axis=1), ddf22.get().reset_index().drop(["author", "value", 'index'], axis=1), ddf21.get().reset_index().drop(["author", "value", 'index'], axis=1), ddf3.get().reset_index().drop(["author", "value", 'index'], axis=1)], axis=1)
    print("time to merge ", time.time() - t1)

    df_res.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/processed_train_post_" + prefix + ".pkl")
    df_res.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/processed_train_post_" + prefix + ".csv")
