#!/usr/bin/env python

import sys
import json
import re
from collections import Counter

black_list_fantasy = ['whowouldwin', 'wouldyourather', 'scenesfromahat', 'AskOuija', 'cosplay', 'cosplaygirls', 'DnD', 'DnDGreentext', 'DnDBehindTheScreen', 'dndnext', 'dungeonsanddragons', 'criticalrole', 'DMAcademy', 'magicTCG', 'modernmagic', 'zombies', 'cyberpunk', 'fantasy', 'scifi', 'starwars', 'startrek', 'asksciencefiction', 'prequelmemes', 'empiredidnothingwrong', 'SequelMemes', 'sciencefiction', 'DarkMatter', 'DefianceTV', 'DoctorWho', 'KilljoysTV', 'OtherSpaceTV', 'RedDwarf', 'StarWarsRebels', 'ThunderbirdsAreGo', 'Andromeda', 'Babylon5', 'Caprica', 'Farscape', 'Firefly', 'Futurama', 'LostInSpace', 'Lexx', 'Space1999', 'SpaceAboveandBeyond', 'SGA', 'DeepSpaceNine', 'StarTrekEnterprise', 'TNG', 'TheAnimatedSeries', 'TOS', 'Voyager', 'TheCloneWars', 'TheThunderbirds', 'LV426', 'BSG', 'Defiance', 'Dune', 'GalaxyQuest', 'DontPanic', 'Spaceballs', 'Stargate', 'Treknobabble', 'StarWars', 'themartian', 'Thunderbirds', 'printSF', 'ScienceFiction', 'SciFi', 'AskScienceFiction', 'movies', 'Television', 'SpaceGameJunkie', 'EliteDangerous', 'StarCitizen', 'AttackWing', 'startrekgames', 'sto', 'gaming', 'Games', 'outside', 'truegaming', 'gamernews', 'gamephysics', 'webgames', 'IndieGaming', 'patientgamers', 'AndroidGaming', 'randomactsofgaming', 'speedrun', 'gamemusic', 'emulation', 'MMORPG', 'gamecollecting', 'hitboxporn', 'gamingcirclejerk', 'gamersriseup', 'gamingdetails', 'gaming4gamers', 'retrogaming', 'GameDeals', 'steamdeals', 'PS4Deals', 'freegamesonsteam', 'shouldibuythisgame', 'nintendoswitchdeals', 'freegamefindings', 'xboxone', 'oculus', 'vive', 'paradoxplaza', 'pcmasterrace', 'pcgaming', 'gamingpc', 'steam', 'linux_gaming', 'nintendo', '3DS', 'wiiu', 'nintendoswitch', '3dshacks', 'amiibo', 'sony', 'PS3', 'playstation', 'vita', 'PSVR', 'playstationplus', 'PS4', 'PS4Deals', 'DotA2', 'starcraft', 'smashbros', 'dayz', 'civ', 'KerbalSpaceProgram', 'masseffect', 'clashofclans', 'starbound', 'heroesofthestorm', 'terraria', 'dragonage', 'citiesskylines', 'smite', 'bindingofisaac', 'eve', 'starcitizen', 'animalcrossing', 'metalgearsolid', 'elitedangerous', 'bloodborne', 'monsterhunter', 'warframe', 'undertale', 'thedivision', 'stardewvalley', 'nomansskythegame', 'totalwar', 'pathofexile', 'ClashRoyale', 'crusaderkings', 'dwarffortress', 'eu4', 'thesims', 'assassinscreed', 'playrust', 'forhonor', 'stellaris', 'kingdomhearts', 'blackdesertonline', 'factorio', 'Warhammer', 'splatoon', 'rimworld', 'Xcom', 'streetfighter', 'paydaytheheist', 'MonsterHunterWorld', 'Seaofthieves', 'cyberpunkgame', 'warhammer40k', 'paladins', 'osugame', 'spidermanps4', 'persona5', 'horizion', 'reddeadredemption', 'mountandblade', 'deadbydaylight', 'farcry', 'hoi4', 'warthunder', 'grandorder', 'divinityoriginalsin', 'escapefromtarkov', 'theexpanse', 'darkestdungeon', 'forza', 'godofwar', 'ark', 'bioshock', 'edh', 'summonerswar', 'duellinks', 'arma', 'pathfinderrpg', 'footballmanagergames', 'kingdomcome', 'subnautica', 'thelastofus', 'doom', 'jrpg', 'borderlands', 'borderlands2', 'DarkSouls', 'DarkSouls2', 'DarkSouls3', 'diablo', 'diablo3', 'elderscrollsonline', 'ElderScrolls', 'teslore', 'Skyrim', 'skyrimmods', 'fallout', 'fo4', 'fo76', 'fireemblem', 'FireEmblemHeroes', 'FortniteBR', 'Fortnite', 'FortniteBattleRoyale', 'Fortnitecompetitive', 'FortniteLeaks', 'GrandTheftAutoV', 'gtav', 'gtaonline', 'hearthstone', 'CompetitiveHS', 'customhearthstone', 'minecraft', 'feedthebeast', 'overwatch', 'competitiveoverwatch', 'overwatchuniversity', 'Overwatch_Memes', 'Overwatch_Porn', 'PUBATTLEGROUNDS', 'PUBG', 'pubgxboxone', 'pubgmobile', 'rocketleague', 'rocketleagueexchange', 'witcher', 'gwent', 'tf2', 'starwarsbattlefront', 'rainbow6', 'titanfall', 'shittyrainbow6', 'battlefield_4', 'battlefield', 'battlefield_one', 'blackops3', 'CODZombies', 'callofduty', 'WWII', 'blackops4', 'codcompetitive', 'GlobalOffensive', 'globaloffensivetrade (private)', 'csgo', 'halo', 'haloonline', 'fifa', 'nba2k', 'DestinyTheGame', 'fireteams', 'destiny2', 'leagueoflegends', 'summonerschool', 'LoLeventVODs', 'wow', 'guildwars2', 'swtor', 'ffxiv', 'FinalFantasy', 'ffxv', 'Pokemon', 'friendsafari', 'pokemontrades', 'pokemongo', 'TheSilphRoad', 'Runescape', '2007scape', 'zelda', 'breath_of_the_wild']
black_list_dating = ['r4r', 'AlienExchange', 'NeedAFriend', 'BDSMPersonals', 'NSFWSkype', 'ChatPals', 'OkCupidProfiles', 'CuddleBuddies', 'OlderWomen', 'dirtyr4r', 'CommittedDating', 'Penpals', 'coffeemeetsbagel', 'ForeverAloneDating', 'POF', 'R4R30Plus', 'PlentyofFish', 'GamerPals', 'tinder', 'RandomActsofBlowjob', 'datingoverthirty', 'randomactsofmuffdive', 'randomactsofmakingout', 'random_acts_of_sex', 'Hardshipmates', 'RedditRoommates', 'InternetFriends', 'KiKPals', 'LetsChat', 'MakeNewFriendsHere', 'hookup', 'OnlineDating', 'atx4atx', 'euro4euro', 'nj4nj', 'br4br', 'satx4satx', 'NeedAFriend', 'NSFWSkype', 'OkCupidProfiles', 'OlderWomen', 'Penpals', 'R4R30Plus', 'RandomActsofBJ', 'RedditRoommates', 'Sex', 'LetsChat', 'RandomActsofMD', 'AlienExchange', 'BDSMPersonals', 'ChatPals', 'CuddleBuddies', 'DirtyR4R', 'ForeverAloneDating', 'GamerPals', 'Hardshipmates', 'InternetFriends', 'KiKPals', 'MakeNewFriendsHere', 'Black4Black']


syn_list = ['nurse', 'astronomer', 'chemist', 'physicist', 'geographer', 'geologist', 'biologist', 'scientist', 'surgeon', 'cleric', 'preacher', 'reverend', 'pastor', 'priest', 'educator', 'lecturer', 'tutor', 'school teacher', 'teacher', 'sportsman', 'swimmer', 'football player', 'tennis player', 'basketball player', 'hockey player', 'baseball player', 'soccer player', 'golf player', 'athlete', 'broker', 'tattooist', 'tattooer', 'tattoo artist', 'investor', 'librarian', 'actress', 'performer', 'thespian', 'actor', 'social worker', 'supervisor', 'manager', 'butcher', 'circus performer', 'jurist', 'counselor', 'attorney', 'barrister', 'lawyer', 'proofreader', 'editor', 'dentist', 'graphic designer', 'cartoonist', 'illustrator', 'physician', 'doctor', 'beautician', 'barber', 'friseur', 'hairdresser', 'mechanic', 'machinist', 'woodworker', 'carpenter', 'delivery man', 'delivery driver', 'receptionist', 'paparazzo', 'photojournalist', 'photographer', 'chauffeur', 'motorist', 'automobilist', 'driver', 'policeman', 'constable', 'gendarme', 'cop', 'policewoman', 'police officer', 'physical trainer', 'athletic trainer', 'sport coach', 'founder', 'innovator', 'startuper', 'entrepreneur', 'advisor', 'consultant', 'cook', 'chef', 'executive chef', 'line chef', 'baker', 'film maker', 'film producer', 'software engineer', 'hacker', 'coder', 'computer scientist', 'programmer', 'head hunter', 'recruiter', 'stewardess', 'steward', 'air hostess', 'cabin crew', 'air host', 'flight attendant', 'painter', 'pupil', 'undergraduate', 'schoolchild', 'undergrad', 'graduate', 'college student', 'school student', 'student', 'assistant', 'secretary', 'bodyguard', 'security guard', 'essayist', 'novelist', 'writer', 'civil engineer', 'architect', 'plumber', 'columnist', 'correspondant', 'reporter', 'journalist', 'banker', 'magician', 'babysitter', 'au pair', 'governess', 'nanny', 'porter', 'cleaner', 'janitor', 'retailer', 'trader', 'vendor', 'shopkeeper', 'merchant', 'retailer', 'lyricist', 'poet', 'salesperson', 'saleswoman', 'salesman', 'pharmacologist', 'apothecary', 'pharmacist', 'sculptor', 'artist', 'psychotherapist', 'psychoanalyst', 'psychologist', 'fisherman', 'sailor', 'scholar', 'postdoc', 'research fellow', 'professor', 'academic', 'singer', 'composer', 'piano player', 'guitar player', 'drum player', 'trumpet player', 'violin player', 'flute player', 'conductor', 'musician', 'aviator', 'aircraft pilot', 'airplane pilot', 'barista', 'property owner', 'hotelkeeper', 'landlord', 'waitress', 'waiter', 'military personnel', 'guerilla', 'army veteran', 'military veteran', 'enlisted man', 'enlisted woman', 'soldier', 'firefighter', 'cheerleader', 'constructor', 'construction worker', 'builder', 'bookkeepeer', 'accountant', 'engineer', 'screenwriter', 'ballet dancer', 'ballerina', 'hip-hop dancer', 'salsa dancer', 'rumba dancer', 'swing dancer', 'belly dancer', 'flamenco dancer', 'latin dancer', 'dancer']

lexicon = [line.strip() for line in open("empath_lex_profession.txt").readlines()]

patterns = ["i am", "i'm", "my profession", "i work as", "my job", "my occupation", "i regret becoming"]

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
        if "\"" in js.get('selftext', '') or "\"" in js.get('body', ''):
            continue
        if js['subreddit'] in black_list_fantasy + black_list_dating:
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
        prof_check = any([" " + itm + " " in txt for itm in syn_list])
        if prof_check:
            pat_check = any([" " + itm + " " in txt for itm in patterns])
            if pat_check:
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
