from utils import *
from label_post import *
from label_user import *
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--load_posts", action="store_true")
parser.add_argument("--label_posts", action="store_true")
parser.add_argument("--load_users", action="store_true")
parser.add_argument("--group_posts", action="store_true")
parser.add_argument("--join", action="store_true")
parser.add_argument("--label_users", action="store_true")
parser.add_argument("--do_all", action="store_true")
parser.add_argument("--month", action="store_true")
parser.add_argument("--year", action="store_true")
parser.add_argument("--all", action="store_true")
args = parser.parse_args()

p = 'all'
p = 'month' if args.month else p
p = 'year' if args.year else p
p = 'all' if args.all else p
print(p)

t0 = time.time()
if args.load_posts:
    process_reddit_post_csv("/home/tigunova/hobby_post_" + p + "/part-00000", prefix = p)
elif args.label_posts:
    label_post("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/raw_train_posts_" + p + ".pkl", prefix = p)
elif args.load_users:
    process_reddit_user_raw("/home/tigunova/hobby_user_" + p + "/part-00000", prefix = p)
elif args.group_posts:
    merge_post_df("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/train_post_" + p + ".pkl", prefix = p)
elif args.join:
    join_post_author("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/processed_train_post_" + p + ".pkl", "/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/train_authors_" + p + ".pkl", prefix = p)
elif args.label_users:
    label_user("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/joint_post_author_" + p + ".pkl", prefix = p)


if args.do_all:
    process_reddit_post_csv("/home/tigunova/hobby_post_" + p + "/part-00000", prefix = p)
    print("load post total time, mins ", (time.time() - t0) / 60)
    t1 = time.time()
    label_post("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/raw_train_posts_" + p + ".pkl", prefix = p)
    print("label posts total time, mins ", (time.time() - t1) / 60)
    t1 = time.time()
    process_reddit_user_raw("/home/tigunova/hobby_user_" + p + "/part-00000", prefix=p)
    print("load users total time, mins ", (time.time() - t1) / 60)
    t1 = time.time()
    merge_post_df("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/train_post_" + p + ".pkl", prefix=p)
    print("group posts total time, mins ", (time.time() - t1) / 60)
    t1 = time.time()
    join_post_author("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/processed_train_post_" + p + ".pkl", "/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/train_authors_" + p + ".pkl", prefix = p)
    print("join post author total time, mins ", (time.time() - t1) / 60)
    t1 = time.time()
    label_user("/home/tigunova/PycharmProjects/snorkel_labels/data/hobby/joint_post_author_" + p + ".pkl", prefix = p)
    print("label users total time, mins ", (time.time() - t1) / 60)
    t1 = time.time()

print("total total time, mins ", (time.time() - t0) / 60)