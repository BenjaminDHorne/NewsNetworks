from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
import numpy as np
import json
from datetime import datetime
import nltk
from collections import Counter
import unicodedata
import networkx as nx
from networkx.algorithms.components.connected import connected_components
from itertools import combinations
import math
from operator import itemgetter
from nltk import word_tokenize

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def fix(text):
    try:
        text = text.decode("ascii", "ignore")
    except:
        t=[unicodedata.normalize('NFKD', unicode(q)).encode('ascii','ignore') for q in text]
        text=''.join(t).strip()
    return text

def missing_data_heuristic(allpairs_in_componenent, perfect_pairs, source, already_done):
    lookup_path = "D:\\RPI Research\\News Producer Networks\\Data\\Plain Text Data\\Content\\"
    for pair in allpairs_in_componenent:
        taken = False
        if pair[0].split("--")[0] != source and pair[0].split("--")[1] != source:
            filesource = pair[0].split("--")[0]
            filedate = pair[0].split("--")[1]
            try:
                with open(lookup_path+filedate+"\\"+filesource+"\\"+pair[0]) as f:
                    text = " ".join([line for line in f])
                tokens = word_tokenize(text)
                if len([1 for token in tokens[:10] if token == "%s"%(source)]) > 0:
                    perfect_pairs.append((source, pair[0]))
                    taken = True
                filesource = pair[1].split("--")[0]
                filedate = pair[1].split("--")[1]
                with open(lookup_path+filedate+"\\"+filesource+"\\"+pair[1]) as f:
                    text = " ".join([line for line in f])
                tokens = word_tokenize(text)
                if len([1 for token in tokens[:10] if token == "%s"%(source)]) > 0:
                    if (source, pair[1]) in already_done or pair[1].strip().split("--")[0] == "Reuters":
                        continue
                    perfect_pairs.append((source, pair[1]))
                    taken = True
            except:
                continue
            if taken:
                print "Removed\n"
                allpairs_in_componenent.remove(pair)
            taken = False
    return allpairs_in_componenent, perfect_pairs

def build_candidate_set(m, day_range, sim_tolerance, folderpath):
    day_start = day_range[0]
    day_end = day_range[1]
    documents = []
    document_labels = []
    candidate_matching_pairs = []
    for dirName, subdirList, fileList in os.walk(folderpath):
        for fn in fileList:
            source = fn.split("--")[0]
            date = fn.split("--")[1]
            month = int(date.split("-")[1])
            day = int(date.split("-")[-1])
            if day >= day_start and day <= day_end and month == m:
                with open(dirName + "//" + fn) as article:
                    text = " ".join([line for line in article])
                documents.append(text)
                document_labels.append(fn)
            else:
                continue

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(documents)
    pairwise_similarity_matrix = tfidf * tfidf.T  # similarity among all article that week
    p = pairwise_similarity_matrix.toarray()

    ii = np.unravel_index(np.argsort(p.ravel()), p.shape)

    for j in xrange(len(ii[0])):
        if document_labels[ii[0][j]].split("--")[0] != document_labels[ii[1][j]].split("--")[0] and p[ii[0][j], ii[1][j]] >= sim_tolerance:
            #print document_labels[ii[0][j]], document_labels[ii[1][j]], p[ii[0][j], ii[1][j]]
            if (document_labels[ii[1][j]], document_labels[ii[0][j]]) not in candidate_matching_pairs:  # inverse pair of articles
                candidate_matching_pairs.append((document_labels[ii[0][j]], document_labels[ii[1][j]]))

    return candidate_matching_pairs


def compute_overlapping_pairs(candidate_matching_pairs): #A connected component problem.
    G = nx.Graph()
    G.add_edges_from(candidate_matching_pairs) #nodes are specific article files
    overlapping_matching_articles = connected_components(G)
    #print list(overlapping_matching_articles)
    return list(overlapping_matching_articles)

def make_perfect_matches(overlapping_matching_articles, already_done):
    perfect_pairs = []
    lookup_path = "D:/RPI Research/News Producer Networks/Data/articles_7232018/new articles/"
    for component in overlapping_matching_articles:
        pair_info = {}
        allpairs_in_componenent = list(combinations(component, 2))

        if len([1 for pair in allpairs_in_componenent if pair not in already_done]) != len(allpairs_in_componenent): #already done componenet
            continue

        for pair in allpairs_in_componenent: #remove same source pairs
            if pair[0].split("--")[0] == pair[1].split("--")[0]:
                print "Removed Same Source Pair", pair[0], pair[1]
                allpairs_in_componenent.remove(pair)

        for pair in allpairs_in_componenent:
            p0 = pair[0]
            p0_date = p0.split("--")[1]
            p0_source = p0.split("--")[0]
            fn_path0 = lookup_path + p0_date + "/" + p0_source + "/" + p0
            with open(fn_path0) as p0_jsonfile:
                x = json.loads(p0_jsonfile.readline())
                p0_published = x['published']
                p0_text = x['content']
            p1 = pair[1]
            p1_date = p1.split("--")[1]
            p1_source = p1.split("--")[0]
            fn_path1 = lookup_path + p1_date + "/" + p1_source + "/" + p1
            with open(fn_path1) as p1_jsonfile:
                x = json.loads(p1_jsonfile.readline())
                p1_published = x['published']
                p1_text = x['content']
            v0 = text_to_vector(p0_text)
            v1 = text_to_vector(p1_text)
            sim = get_cosine(v0, v1)
            try:
                p0_datetime = datetime.strptime(p0_published.split("+")[0], '%Y-%m-%d %H:%M:%S')
                p1_datetime = datetime.strptime(p1_published.split("+")[0], '%Y-%m-%d %H:%M:%S')
            except:
                continue

            pair_info[pair] = [sim, (p0, p0_datetime), (p1, p1_datetime)]

        #filter component
        heuristic_source_list = ["Reuters", "AP"]
        for source in heuristic_source_list:
            if len([1 for pair in allpairs_in_componenent if pair[0].split("--")[0] == source or pair[0].split("--")[1] == source]) == 0:
                    allpairs_in_componenent, perfect_pairs = missing_data_heuristic(allpairs_in_componenent, perfect_pairs, source, already_done) #missing data heuristic
        if len(allpairs_in_componenent) == 0:
            continue
        try:
            if len(allpairs_in_componenent) == 1: #only 1 pair in time frame <----
                if pair_info[allpairs_in_componenent[0]][1][1] < pair_info[allpairs_in_componenent[0]][2][1]: #direction of pair
                    perfect_pairs.append((allpairs_in_componenent[0][0],allpairs_in_componenent[0][1]))
                else:
                    perfect_pairs.append((allpairs_in_componenent[0][1], allpairs_in_componenent[0][0]))

            sources_by_date = [pair_info[p][1] for p in allpairs_in_componenent];[sources_by_date.append(pair_info[p][2]) for p in allpairs_in_componenent]
            sources_by_date = sorted(sources_by_date, key=itemgetter(1))
            sims_in_comp = sorted([pair_info[p][0]  for p in allpairs_in_componenent], reverse=True)
            sims_and_pairs = sorted([(pair_info[p][0], pair_info[p][1], pair_info[p][2])  for p in allpairs_in_componenent], key=itemgetter(0), reverse=True)
            sim_count = Counter(sims_in_comp)
            print sims_in_comp
            if sim_count[sims_in_comp[0]] == 1: #highest similar pair is unique <----
                if sims_and_pairs[0][1][1] < sims_and_pairs[0][2][1]:  # direction of pair
                    perfect_pairs.append((sims_and_pairs[0][1][0], sims_and_pairs[0][2][0]))
                else:
                    perfect_pairs.append((sims_and_pairs[0][2][0], sims_and_pairs[0][1][0]))
                del sims_and_pairs[0]
                del sims_in_comp[0]
            oldest_fn = sources_by_date[0]
            oldest_found = False
            for pair in sims_and_pairs: #take those more unique than oldest, otherwise take oldest <-------
                if oldest_fn in pair:
                    oldest_found = True
                    if pair[1][1] < pair[2][1]:  # direction of pair
                        perfect_pairs.append((pair[1][0], pair[2][0]))
                    else:
                        perfect_pairs.append((pair[2][0], pair[1][0]))
                if oldest_found == False:
                    if pair[1][1] < pair[2][1]:  # direction of pair
                        perfect_pairs.append((pair[1][0], pair[2][0]))
                    else:
                        perfect_pairs.append((pair[2][0], pair[1][0]))
        except:
           print "Key Error..."


    return perfect_pairs

def check_already_done(fn):
    pairs_done = []
    with open(fn) as already:
        for line in already:
            l = line.strip().split(",")
            pairs_done.append((l[0],l[1]))
    return pairs_done

#main
cant_be_first_list = ["Drudge Report", "The Right Scoop", "True Pundit", "Western Journal", "oann"]
outputfile = "Article_Matches_Perfect_MAGAHATKIDS.csv"
already_done = check_already_done(outputfile)
sim_tolerance = 0.80
months = [01]
day_ranges = [(15, 31)]#[(1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 31)]
folderpath = "D:\\RPI Research\\News Producer Networks\\Data\\Plain Text Data\\New Content\\"
for m in months:
    for day_range in day_ranges:
        print "Building Candidate Set for days", day_range, "..."
        candidate_matching_pairs = build_candidate_set(m, day_range, sim_tolerance, folderpath)
        print "Extracting Overlapping Pairs..."
        overlapping_matching_articles = compute_overlapping_pairs(candidate_matching_pairs)
        print "Making Perfect Pairs..."
        perfect_pairs = make_perfect_matches(overlapping_matching_articles, already_done)

        with open(outputfile, "a") as out:
            written = []
            perfect_pairs = list(set(perfect_pairs))
            for p in perfect_pairs:
                if p[0].split("--")[0] in cant_be_first_list:#Heuristic cant be first, aggregators
                    #continue
                    newp = (p[1], p[0])
                    out.write(",".join(newp) + "\n")
                    continue
                if p[1] in written: #article can only copy once in this setting.
                    continue
                out.write(",".join(p)+"\n")
                written.append(p[1])