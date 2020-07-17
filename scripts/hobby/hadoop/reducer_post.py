#!/usr/bin/env python

import sys

current_author = None
current_prof = "-1"
current_texts = []
current_count = 1
current_subs = []
current_ids = []
current_profs = []

for line in sys.stdin:
    line = line.strip()

    if len(line.split('\t')) != 2:
        continue

    try:
        author, stuff = line.split('\t')
        prof, txt = stuff.split("___")
        subr, iid, txt = txt.split("++++")
    except:
        continue

    if current_author == author:
        if current_count <= 52:
            current_count += 1
            if prof != "-1":
                current_texts.append(txt)
                current_subs.append(subr)
                current_ids.append(iid)
                current_profs.append(prof)
                current_prof = prof
    else:
        if current_count > 10 and current_count <= 50 and current_prof != "-1":
            for txtt, subb, idd, prr in zip(current_texts, current_subs, current_ids, current_profs):
                for prrr in prr.split(":::"):
                    print(u'%s,%s,%s,%s,%s' % (current_author.replace(",",""), prrr.replace(",",""), subb.replace(",",""), idd.replace(",",""), txtt.replace(",","")))
        current_author = author
        current_prof = prof
        current_count = 1
        if txt != "":
            current_texts = [txt]
            current_ids = [iid]
            current_subs = [subr]
            current_profs = [prof]
        else:
            current_texts = []
            current_ids = []
            current_subs = []
            current_profs = []