import csv
import random
import Levenshtein as lev
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--month", action="store_true")
parser.add_argument("--year", action="store_true")
parser.add_argument("--all", action="store_true")
args = parser.parse_args()

p = 'year'
p = 'month' if args.month else p
p = 'year' if args.year else p
p = 'all' if args.all else p

thr = 0.9
resulting_dict = eval(open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/final_author_dict_" + p + ".txt").read())
resulting_dict = dict((x[0],[y for y in x[1] if float(y[1]) > thr]) for x in resulting_dict.items() if len(["".join([str(z) for z in y]) for y in x[1] if float(y[1]) > thr]) > 1)

user_file = "/home/tigunova/PycharmProjects/snorkel_labels/data/profession/train_post_" + p + ".csv"
result_file = "/home/tigunova/PycharmProjects/snorkel_labels/data/profession/user_" + p + ".csv"
msg_dict = defaultdict(list)
seen_ids = set()

reader = csv.reader(open(result_file))
all_usr = sorted([x for x in reader if len(x[4].split(",")) > 0 and x[-2] != "neg_prob" and float(x[-1]) > 0.5 and len(x[13].split(",")) > 1], key=lambda y:float(y[-1]), reverse=True)
selected = all_usr[:1000]
#selected = [x for x in reader if len(x[4].split(",")) > 1 and float(x[-2]) > 0.99]
random.shuffle(selected)
usr_head = next(csv.reader(open(result_file)))
selected_usr_set = set([x[2] for x in selected])
all_usr_dict = dict((x[2] + "_" + x[3], [y for y in zip(usr_head, x) if ("more" in y[0] or "prob" in y[0]) and y[-1] != "-1"]) for x in all_usr)
seenies = defaultdict(set)

for line in csv.reader(open(user_file)):
    if line[2] in selected_usr_set and line[2] in resulting_dict:
        if line[5] + "_" + line[3] not in seen_ids:
            seen_ids.add(line[5] + "_" + line[3])
            curr_seenies = seenies[line[2]]
            ses = False
            for seenie in curr_seenies:
                if lev.ratio(seenie[0], line[6].lower()) > 0.85:# and line[3] == seenie[1]:
                    ses = True
            if not ses:
                seenies[line[2]].add((line[6].lower(), line[3]))
                if len(line[6]) > 5:
                    msg_dict[line[2]].append([line[5] + "_" + line[3], line[6]] + line[9:])
                if len(msg_dict) > 100:
                    break

# for usr in selected_usr_set:
#     print(usr, resulting_dict[usr])
#     for msg in msg_dict[usr]:
#         print(msg)
#     print("\n\n")

i = 1
head = next(csv.reader(open(user_file)))
head = [x.replace("LabelingFunction ","") for x in head if "labeling" in x.lower() or "_prob" in x.lower()]
commas = "," * len(head)
for usr, msgs in msg_dict.items():
    print("%d,,,," % (i))
    print(",%s,,," % (usr))
    for msg in msgs:
        if len(msg) > 5:
            #el_fuc = [head[y[0]] + ":" + y[1] for y in enumerate(msg[2:]) if y[1] != "-1"]
            print(",,,%s," %  (",".join([msg[1], msg[0]] + [":".join(y) for y in all_usr_dict.get(usr + "_" + msg[0].split("_")[1], "")])))
    print(",,,," + commas)
    print(",The profession is,,%s," % (repr(resulting_dict[usr]).replace(",", ";")))
    print(",,,," + commas)
    i += 1
print("-1,,,," + commas)
        