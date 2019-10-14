from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from datetime import timedelta
from networkx.algorithms.components.connected import connected_components
from itertools import combinations
from operator import itemgetter

import networkx as nx
import numpy as np
import sqlite3
from threading import Thread
import re


def get_connection(database):
    conn = sqlite3.connect(database)
    return conn

def get_max_timestamp(conn):
    try:
        sql_string = "SELECT max(published_utc) FROM newsdata"
        cursor = conn.cursor()
        cursor.execute(sql_string)
        return int(cursor.fetchone()[0])
    except Exception as e:
        print(e)


def get_articles_per_source(conn):
    try:
        sql_string = "SELECT source,count(id) FROM newsdata GROUP BY source"
        cursor = conn.cursor()
        cursor.execute(sql_string)
        article_count = {str(r[0]):int(r[1]) for r in cursor.fetchall()}
        return article_count
    except Exception as e:
        print(e)


def get_documents(conn, start_timestamp, window_size=4):
    try:
        end_timestamp = int(start_timestamp) + window_size * 86400  # no. of seconds in a day
        sql_string = "SELECT id,source,content,published_utc,author FROM newsdata \
        WHERE published_utc > %d AND published_utc < %d \
        ORDER BY published_utc" % (start_timestamp, end_timestamp)
        cursor = conn.cursor()
        cursor.execute(sql_string)
        r = list(cursor.fetchall())
        print("%d documents" % len(r))

        ids = [s[0] for s in r]
        documents = [s[2] for s in r]
        published_utc = [s[3] for s in r]
        sourcelist = [s[1] for s in r]
        authorlist = [s[4] for s in r]
        sources = dict()
        published = dict()
        documents_dict = dict()
        authors = dict()
        for i, s, p, c, a in zip(ids, sourcelist, published_utc, documents, authorlist):
            sources[i] = s
            published[i] = p
            documents_dict[i] = c
            authors[i] = a

        # ids(list), sources(dict), documents(list), published_utc(dict)
        return ids, sources, documents, published, documents_dict, authors

    except Exception as e:
        print(e)


def candidate_task(t_pool, task, output, sim_threshold, ids, sources):
    for i,j,v in t_pool:
        if v >= sim_threshold and sources[ids[i]] != sources[ids[j]] and i > j:
            output[task].append((ids[i], ids[j], v))

def build_candidate_set(ids, sources, documents, sim_threshold=0.85):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(documents)
    print("Computing similarities...")
    pairwise_sim = tfidf * tfidf.T  # similarity among all article that week
    #get non-zero entries in sparse matrix
    cx = pairwise_sim.tocoo()


    candidate_pairs = list()
    candidate_similarity = dict()
    # run over the upper triangular matrix to find similarities
    # ps: this is a sparse matrix so iterating by izip is MUCH faster
    print("Finding pairs...")
    n_threads = 12
    offset = len(cx.row)//n_threads
    t_pools = [zip(cx.row[i*offset:(i+1)*offset], cx.col[i*offset:(i+1)*offset], \
                            cx.data[i*offset:(i+1)*offset]) for i in range(n_threads)]

    output = [list() for r in range(n_threads)]  # store output from threads before joining
    threads = list()

    for task in range(n_threads):
        t = Thread(target=candidate_task, args=(t_pools[task], task, output, \
                                                    sim_threshold, ids, sources,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Join output from threads
    print("Synchronizing...")
    for r in output:
        for i,j,v in r:
            candidate_pairs.append((i,j,v))
            candidate_similarity[i+j] = v


    print("%d sim pairs." % len(candidate_pairs))
    return candidate_pairs, candidate_similarity


# Given connected component g, select pairs using tree heuristic
def select_most_correct_pairs(c, G, sources):
    selected_pairs = list()
    for node in c:
        e = G.edges(node)
        e = sorted(e, key=lambda u: G.get_edge_data(u[0], u[1])["weight"], reverse=True)
        e = [ed for ed in e if G.nodes[ed[1]]["published"] < G.nodes[ed[0]]["published"]]

        if len(e) > 0:
            first = G.get_edge_data(e[0][0], e[0][1])["weight"]
            ties = list()
            i = 0
            while  i < len(e) and G.get_edge_data(e[i][0], e[i][1])["weight"] == first:
                if sources[node] != sources[e[i][1]]:
                    ties.append(e[i][1])
                i+=1

            selected = min(ties, key=lambda t: G.nodes[t]["published"])

            selected_pairs.append((selected,node))

    return selected_pairs


def compute_overlapping_pairs(candidate_pairs, published, sources):
    print("Overlapping pairs")
    G = nx.Graph()
    G.add_weighted_edges_from(candidate_pairs)
    nx.set_node_attributes(G, published, name="published")
    cc = connected_components(G)

    selected_pairs = list()
    for c in cc:
        selected_pairs.extend(select_most_correct_pairs(c, G, sources))

    return selected_pairs


def missing_data_heuristic(selected_pairs, documents_dict, sources):
    missing_sources = ["AP", "Reuters"]

    regex_list = list()
    # compile regex for news wire
    for m in missing_sources:
        regex_list.append(re.compile(".*(%s).*"%m))

    updated_pairs = list()
    for pair in selected_pairs:
        for m, r in zip(missing_sources, regex_list):
            match0 = r.match(documents_dict[pair[0]])
            match1 = r.match(documents_dict[pair[1]])

            if match0 is None and match1 is None:
                updated_pairs.append(pair)
                break
            else:
                if sources[pair[0]] != m or sources[pair[1]] != m:
                    updated_pairs.append((m,pair[0]))
                    updated_pairs.append((m,pair[1]))
                    break
    return updated_pairs


def author_heuristic(selected_pairs, authors):
    source_names = dict()
    ap_names=["AP", "Associated Press", "The Associated Press", "AP Reports"]
    zerohedge_names = ["Zero Hedge", "ZeroHedge"]
    rt_names = ["RT"]
    reuters_names = ["Reuters"]

    for a in ap_names:
        source_names[a] = "AP"
    for z in zerohedge_names:
        source_names[z] = "ZeroHedge"
    for r in rt_names:
        source_names[r] = "RT"
    for reu in reuters_names:
        source_names[reu] = "Reuters"

    keys = source_names.keys()
    updated_pairs = list()
    for i in range(len(selected_pairs)):
        pair = selected_pairs[i]

        if pair[0] in keys or pair[1] in keys:  # if pair already contains AP or Reuters we skip it
            updated_pairs.append(pair)
        elif pair[0] not in keys and pair[1] not in keys:
            updated_pairs.append(pair)
        else:
            if authors[pair[0]] in keys:
                m = source_names[authors[pair[0]]]
            elif authors[pair[1]] in keys:
                m = source_names[authors[pair[1]]]

            updated_pairs.append((m, pair[0]))
            updated_pairs.append((m, pair[1]))

    return updated_pairs


def aggregrator_heuristic(selected_pairs, sources):
    cant_be_first_list = set(["Drudge Report","theRussophileorg"
     "oann", "Mail", "Yahoo News"])

    for i in range(len(selected_pairs)):
        pair = selected_pairs[i]

        if sources[pair[0]] in cant_be_first_list:
            selected_pairs[i] = (pair[1], pair[0])

    return selected_pairs



def build_network(all_pairs, article_count_per_source, path):
    G = nx.DiGraph()

    weight_dict = {}

    for pair in all_pairs:
        source0 = pair[0].split("--", 1)[0]
        source1 = pair[1].split("--", 1)[0]
        e = (source0,source1)
        if e in weight_dict:
            weight_dict[e] += 1
        else:
            weight_dict[e] = 1

        weight_edges = [(key[0], key[1], float(weight_dict[key])/article_count_per_source[key[1]])\
                                        for key in weight_dict.keys()]

        G.add_weighted_edges_from(weight_edges)

        nx.write_gexf(G, "%s.gexf" % path)




def main():
    pair_file_path = "perfect_pairs.csv"
    conn = get_connection("/media/6gbvolume/nela_eng.db")
    initial_date = "2018-01-01"
    end_timestamp = get_max_timestamp(conn)

    start_date = datetime.strptime(initial_date, "%Y-%m-%d")
    dtime = timedelta(days=4)
    all_pairs = list()
    while True:
        int_date = int(start_date.strftime("%s"))
        if int_date >= end_timestamp:
            break

        ids,sources,documents,published,documents_dict,authors = get_documents(conn, int_date, window_size=4)

        if len(ids) == 0:
            start_date += dtime
            continue


        pairs, simi = build_candidate_set(ids,sources,documents)
        selected_pairs = compute_overlapping_pairs(pairs, published, sources)

        selected_pairs = aggregrator_heuristic(selected_pairs, sources)
        selected_pairs = missing_data_heuristic(selected_pairs, documents_dict, sources)
        selected_pairs = author_heuristic(selected_pairs, authors)

        all_pairs.extend(selected_pairs)
        with open(pair_file_path, "a") as fout:
            for p in selected_pairs:
                print("%s,%s" % (p[0], p[1]), file=fout)

        start_date += dtime

    # Collected all pairs, build network
    # Get articles per source
    article_count_per_source = get_articles_per_source(conn)
    build_network(all_pairs, article_count_per_source, path="FullNetwork")




if __name__ == "__main__":
    main()
