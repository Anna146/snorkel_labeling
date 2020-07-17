import csv

syn_dict = dict()
subr_dict = dict()
reader = csv.reader(open("/home/tigunova/Downloads/labeling hobbies - profession list.csv"))

for line in reader:
    if line[0] != "":
        syn_dict[line[0]] = [x.lower().strip() for x in line[2].strip('"').split(",") if x != '']
        syn_dict[line[0]].append(line[0])
        subr_dict[line[0]] = set([x.lower().strip().replace("/r/", "").replace("r/", "").replace("/", "") for x in
                                 line[4].strip('"').split(" ") if x != ''])

with open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_subreddits.txt", 'w') as f:
    f.write(repr(subr_dict))

with open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_synonyms.txt", 'w') as f:
    f.write(repr(syn_dict))

with open("/home/tigunova/PycharmProjects/snorkel_labels/data/profession/sources/profession_list.txt", "w") as f:
    f.write("\n".join(sorted(list(syn_dict.keys()))))

print([itm for l in syn_dict.values() for itm in l])
print(len([itm for l in syn_dict.values() for itm in l]))
print("#######################################")
print([itm for l in subr_dict.values() for itm in l])
print(len([itm for l in subr_dict.values() for itm in l]))
