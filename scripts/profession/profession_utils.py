import os

import numpy as np
import pandas as pd
import swifter
import _pickle as pickle
import base64
import time

ABSTAIN = -1
NEG = 0
POS = 1

similarities_dict = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/similarities_empath.txt").read())

lexicon = [line.strip() for line in open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/empath_lex.txt").readlines()]

hobby_to_syn = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_synonyms.txt").read())
syn_to_hob = dict((syn, val) for val, syns in hobby_to_syn.items() for syn in syns)

def process_reddit_post_csv(data_path, prefix = ""):
    train_set = pd.read_csv(data_path, header=None, names = ["author", "value", "subr", "idd", "text"])
    train_set.drop(columns=["idd"])
    bad_subs = set([x for x in train_set.subr if
                    any(y in x for y in ["meet", "penpals"]) or x in ["chat", "gonewild", "lgbteens", "needafriend", "makenewfriendshere"]])
    train_set = train_set[~train_set.subr.isin(bad_subs)]
    train_set.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/raw_train_posts_" + prefix + ".pkl")
    train_set.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/raw_train_posts_" + prefix + ".csv")

def process_reddit_user_raw(data_path, prefix=""):
    t0 = time.time()
    author_hob_dict = eval(
        open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/author_profession_dict_" + prefix + ".txt").read())
    hobby_empath_dict = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/empath_dict.txt").read())
    #hobby_subred_dict = eval(
    #    open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_subreddits.txt").read())
    empath_list = dict((y, str(x)) for x,y in enumerate([line.strip() for line in
                   open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/empath_lex.txt").readlines()]))
    hobby_empath_dict_inverse = dict((k,[empath_list[x.strip()] for x in v]) for k,v in hobby_empath_dict.items())
    train_set = pd.read_csv(data_path, header=None, names=["author", "words", "counts"])
    train_set = train_set.drop(["counts"], axis=1)
    old_len = train_set.shape
    train_set = train_set[train_set.author.isin(author_hob_dict)]
    train_set['words'] = train_set['words'].swifter.apply(lambda x: pickle.loads(base64.b64decode(x.strip(), "-_")))
    #train_set['subreddits'] = train_set['subreddits'].swifter.apply(lambda x: pickle.loads(base64.b64decode(x.strip(), "-_")))
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
    train_set['lexicon_counts'] = train_set[['words', 'root_value']].swifter.set_dask_threshold(
        dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: sum(
        [x['words'][y] * similarities_dict[x['root_value']][int(y)] for y in x['words'] if
         int(y) in similarities_dict[x['root_value']]]) / max(
        sum([x['words'][y] for y in x['words'] if int(y) in similarities_dict[x['root_value']]]), 1), axis=1)
    print("calculated sims")
    print("time for lex", time.time() - t1)
    train_set = train_set.drop(["words"], axis=1)
    #train_set['subreddit_counts'] = train_set[['subreddits', 'value', 'root_value']].swifter.apply(lambda x: sum(x['subreddits'][y] for y in hobby_subred_dict[x['root_value']] if y in x['subreddits']) +                                                                                         sum(x['subreddits'][y] for y in x['subreddits'] if "_".join(x['value'].split(" ")) or "".join(x['value'].split(" ")) in y or "_".join(x['root_value'].split(" ")) or "".join(x['root_value'].split(" ")) in y), axis=1)
    #train_set['name_in_subr_count'] = train_set[['subreddits', 'value', 'root_value']].swifter.apply(lambda x: sum(x['value'] in y or x['root_value'] in y for y in x['subreddits']), axis=1)
    train_set = train_set.drop(["root_value"], axis=1)
    #train_set = train_set.drop(["subreddits"], axis=1)
    print("initially author-values ", len(author_hob_dict), " num authors in train set ", old_len, "  written records ", train_set.shape)
    print("total time mins", (time.time() - t0) / 60)
    train_set.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/train_authors_" + prefix + ".pkl")
    train_set.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/train_authors_" + prefix + ".csv")

def _process_reddit_user_raw(data_path, prefix=""):
    t0 = time.time()
    author_hob_dict = eval(
        open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/author_profession_dict_" + prefix + ".txt").read())
    hobby_empath_dict = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/empath_dict.txt").read())
    #hobby_subred_dict = eval(
    #    open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_subreddits.txt").read())
    empath_list = dict((y, str(x)) for x,y in enumerate([line.strip() for line in
                   open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/empath_lex.txt").readlines()]))
    hobby_empath_dict_inverse = dict((k,[empath_list[x.strip()] for x in v]) for k,v in hobby_empath_dict.items())
    train_set = pd.read_csv(data_path, header=None, names=["author", "words", "counts"])
    train_set = train_set.drop(["counts"], axis=1)
    old_len = train_set.shape
    train_set = train_set[train_set.author.isin(author_hob_dict)]
    train_set['words'] = train_set['words'].swifter.apply(lambda x: pickle.loads(base64.b64decode(x.strip(), "-_")))
    #train_set['subreddits'] = train_set['subreddits'].swifter.apply(lambda x: pickle.loads(base64.b64decode(x.strip(), "-_")))
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
    train_set['words'] = train_set[['words', 'root_value']].swifter.set_dask_threshold(dask_threshold=0.001).allow_dask_on_strings().apply(lambda x: [x['words'][y] for y in hobby_empath_dict_inverse[x['root_value']] if y in x['words']], axis=1)
    train_set['lexicon_counts'] = train_set['words'].swifter.apply(lambda x: sum(x), axis=1)
    train_set['unique_lexicon_counts'] = train_set['words'].swifter.apply(lambda x: len(x), axis=1)
    print("time for lex", time.time() - t1)
    train_set = train_set.drop(["words"], axis=1)
    #train_set['subreddit_counts'] = train_set[['subreddits', 'value', 'root_value']].swifter.apply(lambda x: sum(x['subreddits'][y] for y in hobby_subred_dict[x['root_value']] if y in x['subreddits']) +                                                                                         sum(x['subreddits'][y] for y in x['subreddits'] if "_".join(x['value'].split(" ")) or "".join(x['value'].split(" ")) in y or "_".join(x['root_value'].split(" ")) or "".join(x['root_value'].split(" ")) in y), axis=1)
    #train_set['name_in_subr_count'] = train_set[['subreddits', 'value', 'root_value']].swifter.apply(lambda x: sum(x['value'] in y or x['root_value'] in y for y in x['subreddits']), axis=1)
    train_set = train_set.drop(["root_value"], axis=1)
    #train_set = train_set.drop(["subreddits"], axis=1)
    print("initially author-values ", len(author_hob_dict), " num authors in train set ", old_len, "  written records ", train_set.shape)
    print("total time mins", (time.time() - t0) / 60)
    train_set.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/train_authors_" + prefix + ".pkl")
    train_set.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/train_authors_" + prefix + ".csv")

def join_post_author(post_path, author_path, prefix = ""):
    posts = pd.read_pickle(post_path)
    authors = pd.read_pickle(author_path)
    print("the length of posts", len(set(posts['author'] + "+++++" + posts['value'])))
    print("the length of authors", len(set(authors['author'] + "+++++" + authors['value'])))
    resulting = pd.merge(posts, authors, on=['author', 'value'], how='inner')
    print("the length of joint", len(set(resulting['author'] + "+++++" + resulting['value'])))
    resulting.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/joint_post_author_" + prefix + ".pkl")
    resulting.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/joint_post_author_" + prefix + ".csv")

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

    df_res.to_pickle("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/processed_train_post_" + prefix + ".pkl")
    df_res.to_csv("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/processed_train_post_" + prefix + ".csv")

