#!/usr/bin/env python

import sys
import json
import re
from collections import Counter

lexicon = [line.strip() for line in open("empath_lex.txt").readlines()]

syn_list = ['candle making', 'tramping', 'wandering', 'hiking', 'lego', 'handball', 'collecting mushrooms', 'mushroom hunting', 'laser tag', 'swimming', 'equestreanism', 'steeplechasing', 'harness racing', 'horse racing', 'rodeo', 'cowboy polo', 'horseback riding', '3d printing', 'model construction', 'model building', 'wood carving', 'woodturning', 'carpentry', 'woodworking', 'coffee roasting', 'basketball', 'amateur radio', 'crossword', 'driving a car', 'driving a motorcycle', 'driving car', 'driving motorcycle', 'driving a moped', 'driving motorcycles', 'driving moped', 'driving cars', 'juggling', 'chess', 'crocheting', 'decorative arts', 'graphic arts', 'decoration', 'graphic design', 'fishing', 'book restoration', 'stone collecting', 'marathon', 'long-distance running', 'short-distance running', 'jogging', 'tramping', 'backpacking', 'baking', 'airplanes', 'lacrosse', 'skydiving', 'blacksmithing', 'cosplaying', 'mineralogy', 'mineral collecting', 'embroidery', 'drama performance', 'comedy performance', 'theatre', 'urban exploration', 'badminton', 'tourism', 'vacationing', 'piligrimage', 'going on a trip', 'going on vacation', 'traveling', 'mountain climbing', 'via ferrata', 'alpinism', 'mountaineering', 'tai chi', 'reading books', 'reading magazines', 'reading news', 'reading stories', 'books', 'literature', 'rugby', 'milling', 'grinding metal', 'filing metal', 'welding', 'brazing', 'soldering', 'riveting', 'metalworking', 'canoeing', 'figure skating', 'tour skating', 'ice skating', 'model aircraft', 'darts', 'chasing animals', 'shooting animals', 'shooting birds', 'hunting birds', 'hunting animals', 'kegling', 'bowling', 'musicals', 'karaoke', 'cabaret', 'singing', 'beach volleyball', 'debating', 'debate', 'coding', 'developing programs', 'programming', 'sculpting', 'boxing', 'straight pool', 'balkline', 'carom', 'balk pool', 'snooker', 'billiards', 'do it yourself', 'repairing', 'diy', 'acrobatics', 'drawing pictures', 'scetching', 'painting', 'writing stories', 'writing poems', 'writing novels', 'writing books', 'poker', 'golfing', 'birding', 'ornitilogy', 'bird watching', 'macrame', 'mountain biking', 'glassblowing', 'roller skating', 'motor racing', 'kart racing', 'auto racing', 'sewing', 'aerobatics', 'air rallies', 'aeromodelling', 'air racing', 'ballooning', 'drone racing', 'general aviation', 'gliding', 'hang gliding', 'human powered aircraft', 'parachuting', 'paragliding', 'power kites', 'air sports', 'parkour', 'canyoneering', 'lumberjack', 'scrambling', 'top roping', 'bouldering', 'hockey', 'ice hockey', 'paintball', 'slacklining', 'sudoku', 'hangman', "rubik's cube", 'solving puzzles', 'doing puzzles', 'playing puzzles', 'rock climbing', 'paddling', 'hydrocycle', 'boat racing', 'rowing', 'entomology', 'insect collecting', 'seashell collecting', 'flower collecting', 'astrophysics', 'astronomy', 'working with ceramics', 'working with porcelain', 'pottery', 'soap making', 'weather prediction', 'weather forecasting', 'meteorology', 'quilting', 'cooking', 'cricket', 'archery', 'scouting', 'camping', 'art collecting', 'rope climbing', 'pommel horse', 'vaulting', 'gymnastics', 'triathlon', 'horoscopes', 'astrology', 'wrestling', 'kayaking', 'antiquities', 'ping pong', 'table tennis', 'origami', 'card collecting', 'genealogy', 'figure skating', 'road biking', 'record collecting', 'baseball', 'water polo', 'taekwondo', 'calligraphy', 'weight training', 'weightlifting', 'powerlifting', 'bodybuilding', 'farming', 'planting growing trees', 'growing flowers', 'growing fruits', 'growing vegetables', 'gardening', 'brewing', 'cryptanalysis', 'cryptography', 'snowboarding', 'snorkeling', 'scuba diving', 'foreign languages', 'language studing', 'learning languages', 'language learning', 'biking', 'riding a bike', 'cycling', 'coloring', 'antique cars', 'collecting cars', 'vintage cars', 'cheerleading', 'judo', 'yoga', 'taking photos', 'making pictures', 'taking pictures', 'photojournalism', 'photography', 'films', 'cinema', 'cinematography', 'movies', 'sand art', 'going to restaurants', 'going to cafe', 'eating out', 'aquascaping', 'playing squash', 'orienteering', 'volleyball', 'mindfullness', 'introspection', 'meditation', 'skateboarding', 'scrapbooking', 'fashion', 'ultimate frisbee', 'ski jumping', 'skiing', 'rafting', 'waltz', 'samba', 'rumba', 'pasadoble', 'tango', 'quickstep', 'foxtrot', 'cha cha', 'disco', 'hustle', 'ballet', 'salsa', 'samba', 'twist', 'hip-hop', 'breakdance', 'flamenco', 'dancing', 'electronics', 'numismatics', 'coin collecting', 'yachting', 'boat cruising', 'sailing', 'animation', 'manga', 'anime', 'philately', 'stamp collecting', 'computer games', 'video games', 'gaming', 'tennis', 'buying clothes', 'buying shoes', 'buying bags', 'shopping', 'book collecting', 'playing piano', 'playing guitar', 'playing violin', 'playing bass', 'playing flute', 'playing drumms', 'playing trumpet', 'plaing cello', 'playing saxophone', 'playing harp', 'music', 'knitting', 'playing monopoly', 'scrabble', 'playing uno', 'jigsaw', 'dnd', 'the game of go', 'mahjong', 'snakes and ladders', 'tick tack toe', 'battleship', 'jenga', 'ludo', 'backgammon', 'checkers', 'skip-bo', 'board games', 'aquariums', 'fish keeping', 'quilling']

black_list_dating = ['r4r', 'AlienExchange', 'NeedAFriend', 'BDSMPersonals', 'NSFWSkype', 'ChatPals', 'OkCupidProfiles', 'CuddleBuddies', 'OlderWomen', 'dirtyr4r', 'CommittedDating', 'Penpals', 'coffeemeetsbagel', 'ForeverAloneDating', 'POF', 'R4R30Plus', 'PlentyofFish', 'GamerPals', 'tinder', 'RandomActsofBlowjob', 'datingoverthirty', 'randomactsofmuffdive', 'randomactsofmakingout', 'random_acts_of_sex', 'Hardshipmates', 'RedditRoommates', 'InternetFriends', 'KiKPals', 'LetsChat', 'MakeNewFriendsHere', 'hookup', 'OnlineDating', 'atx4atx', 'euro4euro', 'nj4nj', 'br4br', 'satx4satx', 'NeedAFriend', 'NSFWSkype', 'OkCupidProfiles', 'OlderWomen', 'Penpals', 'R4R30Plus', 'RandomActsofBJ', 'RedditRoommates', 'Sex', 'LetsChat', 'RandomActsofMD', 'AlienExchange', 'BDSMPersonals', 'ChatPals', 'CuddleBuddies', 'DirtyR4R', 'ForeverAloneDating', 'GamerPals', 'Hardshipmates', 'InternetFriends', 'KiKPals', 'MakeNewFriendsHere', 'Black4Black']

patterns = ["obsessed with", "fond of", "keen on", 'like', 'enjoy', 'love', 'play', 'take joy in', 'adore', 'appreciate', "fan of", 'fascinated by', "interested in", 'fancy', 'mad about', "practise", "sucker for", "hate", "dislike", "detest", "can't stand", "interest", "hobby", "passion", "obsession"]

pronouns = ["i", "i'm", "i've", "me", "my"]

for line in sys.stdin:
    try:
        # 0 load json
        try:
            js = json.loads(line)
        except:
            continue
        if not isinstance(js, dict):
            continue
        if 'author' not in js or 'subreddit' not in js or js['author'] == '[deleted]' or js['author'] == "":
            continue
        if 'selftext' not in js and 'body' not in js:
            continue
        if len((js.get('selftext', '') + js.get('body', '')).strip()) == 0:
            continue
        if 'subreddit' not in js and 'subreddit_id' not in js:
            continue
        if js['subreddit'] in black_list_dating:
            continue
        if "\"" in js.get('selftext', '') or "\"" in js.get('body', ''):
            continue

        # 1 extract text
        txt = (js.get('selftext', '') + ' ' + js.get('body', '') + ' ' + js.get('title', '')) #
        original_txt = txt
        txt = txt.lower()

        pattern = re.compile('#[\w]*')
        txt = pattern.sub(' ', txt)
        pattern = re.compile('[\w]*@[\w]*')
        txt = pattern.sub(' ', txt)
        txt = re.sub(r'https?:\/\/.*[\r\n]*', ' ', txt)

        pattern = re.compile('([^\s\w\']|_)+|\d|\t|\n')
        txt = pattern.sub(' ', txt)
        pattern = re.compile('\s+')
        txt = pattern.sub(" ", txt)
        pattern = re.compile('\.+')
        txt = pattern.sub(" ", txt)
        txt = " " + txt + " "

        # 3 check post len
        post_len = len([x for x in txt.split(" ") if len(x.strip().strip(".")) > 1])
        if post_len > 40 or post_len < 10:
            continue

        # 4 check hobby name
        good = 0
        hob_check = any([" " + itm + " " in txt for itm in syn_list])
        if hob_check:
            pat_check = any([" " + itm + " " in txt for itm in patterns])
            if pat_check:
                pron_check = any([" " + itm + " " in txt for itm in pronouns])
                if pron_check:
                    good = 1

        # 5 check lexicon words
        txt = txt.split(" ")
        bigrams = [txt[i] + " " + txt[i+1] for i in range(0, len(txt) - 1)]
        txt.extend(bigrams)
        txt = Counter(txt)
        subr = str(js["subreddit"].replace("_", ""))
        for j, wrd in enumerate(lexicon):
            if wrd in txt:
                print(u"%s\t%s___%s___%s___%s___%s" % (js["author"], str(good), str(j), txt[wrd], str(js['id']), subr))
        print(u"%s\t%s___%s___%s___%s___%s" % (js["author"], str(good), str(-1), str(-1), str(js['id']), subr))
    except:
        continue
