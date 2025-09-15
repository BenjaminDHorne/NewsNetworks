
import cupy as cp
import numpy as np
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.metrics import pairwise_distances
import cudf as cd
from cudf import Series
import pandas as pd
import csv

from collections import defaultdict


def vectorize_documents(documents):
    """
    Perform TF-IDF Vectorization of an input list of `documents`.

    Args:
        documents (list[str]) : The input list of documents to be vectorized
    """

    # Convert to cudf series
    documents = Series(documents)

    print(documents)

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(dtype=cp.float32)

    # Fit and transform the documents
    x = vectorizer.fit_transform(documents)

    return x


def compute_tf_idf_similarity(x):
    """
    Compute and return the pairwise (cosine) distances.
    """
    d = 1 - pairwise_distances(x, metric='cosine')
    return d


def get_document_copy_matrix(documents, dates, threshold=0.85):
    """
    Cluster documents using TF_IDF distances.

    Args:
        documents (list[str]) : the input list of documents as strings.
        threshold (float) : This is the affinity threshold for the cosine similarity. Documents with similarity at least this value will be treated as "copies" of one another.
    
    Returns:
        d_mask (array[bool]) : Returns a binary mask of copy articles. d_mask[i][j] = True if articles i and j are nearly identical.
    """

    x_tfidf = vectorize_documents(documents)
    a = compute_tf_idf_similarity(x_tfidf)  # get tf-idf matrix
    d_mask = a >= threshold
    e = get_early_document_matrix(dates)  # get matrix that says `True` if article i was published before article j

    print(f"e = \n {e}")

    # return (cp.asnumpy(d_mask) & e)
    return (d_mask & e), a


def get_early_document_matrix(dates):
    """
    Returns a binary mask of documents e where e[i,j] is `True` if document `i` was published before document `j`.

    Args:
        dates (list[datetime.datetime]) : The input list of document datetimes with the publication dates.
    
    Returns:
        e (array[bool]) : A 2D array containing earlier vs. later.
    """

    # e = dates.to_numpy() < dates.to_numpy().reshape(len(dates),1)
    dates_cp = cp.array(dates.astype(cp.int64))
    e = dates_cp > dates_cp.reshape(len(dates_cp), 1)
    return e


if __name__ == "__main__":
    # Sample documents
    file = "~/projects/data/nela-2024.csv"
    file = "nela-recent.csv"
    # file = 'test2.csv'

    df = pd.read_csv(file)
    df['content'] = df['content'].fillna('')

    # df[:5].to_csv("test2.csv")

    df['date_published'] = pd.to_datetime(df['date_published'])
    # df['content'] = df['content'].to_string()
    groups = df.groupby(pd.Grouper(key='date_published', axis=0, freq='3D', sort=True))

    fout_articles = open("article_pairs.csv", "w", encoding='utf-8')
    fout_source_edges = open('source_edges.csv', "w", encoding='utf-8')

    source_pairs = defaultdict(int)
    
    for name, g in groups:
        print(f">>> {name}")
        # print(group)

        documents, dates = g['content'], g['date_published']
        sources = g['name']
        article_ids = g['article_id']
        urls = g['url']
        

        print(dates.to_numpy())

        d, tfidf = get_document_copy_matrix(documents, dates)


        a_writer = csv.writer(fout_articles)
        s_writer = csv.writer(fout_source_edges)
        for i, m in enumerate(d):
            print(article_ids.iloc[i], documents.iloc[i][:20])
            copies = cp.asnumpy(cp.argwhere(m))
            for cp_idx in copies:
                print(cp.asnumpy(cp_idx[0]), f"{article_ids.iloc[cp_idx[0]]} - {urls.iloc[cp_idx[0]]} - {tfidf[i][cp_idx[0]]} - {dates.iloc[cp_idx[0]]} - {documents.iloc[cp_idx[0]][:20]}")


                a_writer.writerow((article_ids.iloc[i], article_ids.iloc[cp_idx[0]]))
                # s_writer.writerow((sources.iloc[i], sources.iloc[cp_idx[0]]))
                if sources.iloc[i] == sources.iloc[cp_idx[0]]:
                    continue
                source_pairs[(sources.iloc[i], sources.iloc[cp_idx[0]])] += 1

    
    for key, value in source_pairs.items():
        s_writer.writerow([*key, value])

    fout_articles.close()
    fout_source_edges.close()


