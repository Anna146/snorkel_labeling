#!/usr/bin/env python

import sys
from collections import Counter
import base64

try:
    import cPickle as pickle
except:
    import _pickle as pickle

current_author = None
current_ids = set()
current_words = dict()
current_good = False
current_subreddits = []

for line in sys.stdin:
    line = line.strip()

    if len(line.split('\t')) != 2:
        continue

    try:
        author, stuff = line.split('\t')
        good, wrd, cnt, iid, subr = stuff.split("___")
        cnt = int(cnt)
        good = int(good)
    except:
        continue

    if current_author == author:
        if len(current_ids) <= 52:
            if iid not in current_ids and subr != "":
                current_subreddits.append(subr)
            current_ids.add(iid)
            current_good = True if good else current_good
            nw = current_words.get(wrd, 0)
            current_words[wrd] = cnt + nw
    else:
        if len(current_ids) > 10 and len(current_ids) <= 50 and current_good:
            #print(u'%s,%s,%s' % (current_author.replace(",",""), '"' + repr(current_words) + '"', '"' + repr(dict(Counter(current_subreddits))) + '"'))
            #print(repr(pickle.dumps({"author":current_author.replace(",",""), "words":current_words, "subreddits":dict(Counter(current_subreddits))}, protocol=2).encode("base64")))
            print(u'%s,%s,%s' % (current_author.replace(",",""), base64.b64encode(pickle.dumps(current_words, protocol=2), "-_"), base64.b64encode(pickle.dumps(dict(Counter(current_subreddits)), protocol=2), "-_")))
        current_author = author
        current_ids = {iid}
        current_good = True if good else 0
        current_words = dict()
        current_words[wrd] = cnt
        current_subreddits = [subr] if subr != "" else []
