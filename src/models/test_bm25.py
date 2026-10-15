"""
This script is used to test the BM25 model on the test data.
"""
import json
import os
from rank_bm25 import BM25Okapi
from sklearn.metrics import average_precision_score, ndcg_score
from .. import config


def load_data(dataset_path: str) -> list:
    """
    Load the data from the dataset
    :param dataset_path: the path to the dataset
    :return: the data in list format
    """
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def test_ranking_per_test_case(test_data: dict) -> dict:
    """
    Test the BM25 model on a single test case
    :param test_data: the test data in dictionary format
    :return: the result of the test in dictionary format
    """
    result = {}
    # get the query and the positive and negative documents
    query = (test_data['query']['title'] or " ") + " " + (test_data['query']['abstract'] or " ")
    # positives
    positives = [(pos['title'] or " ") + " " + (pos['abstract'] or " ") for pos in test_data['pos']]
    pos_len = len(positives)
    # negatives
    negatives = [(neg['title'] or " ") + " " + (neg['abstract'] or " ") for neg in test_data['neg']]
    neg_len = len(negatives)

    # generate the tokenized corpus from the positive and negative documents
    tokenized_corpus = [doc.split(" ") for doc in positives + negatives]

    # inti the bm25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # tokenize the query
    tokenized_query = query.split(" ")

    # get the scores
    doc_scores = bm25.get_scores(tokenized_query)

    # MAP
    is_relevant = [True] * pos_len + [False] * neg_len
    result['MAP'] = average_precision_score(is_relevant, doc_scores)

    # NDCG@10
    relevance = [1 if rel else 0 for rel in is_relevant]
    result['NDCG@10'] = ndcg_score([relevance], [doc_scores], k=10)

    # MRR@10
    sim_score_argsort = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
    mrr_score = 0  # initialize the mrr score
    mrr_at_k = 10  # the rank at which we want to calculate the mrr

    # iterate over the top k documents
    for rank, index in enumerate(sim_score_argsort[0:mrr_at_k]):
        if is_relevant[index]:
            # the rank is 0-based index, so we need to add 1
            mrr_score = 1 / (rank + 1)
            break

    result['MRR@10'] = mrr_score

    # rank first relevant
    rfr_score = len(positives) + len(negatives) + 1  # worst score and outside possibility
    for rank, index in enumerate(sim_score_argsort, 1):
        if is_relevant[index]:
            if rank < rfr_score:  # min rank
                rfr_score = rank
    result['RFR'] = rfr_score

    return result


def test(test_data_path: str):
    """
    Test the BM25 model on the test data
    :param test_data_path: the path to the test data
    :return: None
    """
    OUTPUT_PATH: str = f"{config.Config.OUTPUT_PATH}/ranking/bm25/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    whole_sample_results_file_path = OUTPUT_PATH + 'bm25_whole_sample_results.csv'
    avg_results_file_path = OUTPUT_PATH + 'bm25_results.csv'
    # create csv file
    with open(whole_sample_results_file_path, "w") as f:
        f.write("epoch,sample_nr,MAP,MRR@10,RFR,NDCG@10\n")

    test_data = load_data(test_data_path)
    results = []
    for index, test_case in enumerate(test_data):
        result = test_ranking_per_test_case(test_case)
        results.append(result)
        # write the result to the file
        with open(whole_sample_results_file_path, "a") as f:
            f.write(f"{-1},{index+1},{result['MAP']},{result['MRR@10']},{result['RFR']},{result['NDCG@10']}\n")

    # calculate the MAP
    mean_average_precision = round(sum([result['MAP'] for result in results]) / len(results), 4)
    print(f"MAP = {mean_average_precision}")
    # calculate the MRR@10
    mrr = round(sum([result['MRR@10'] for result in results]) / len(results), 4)
    print(f"MRR@10 = {mrr}")
    # calculate the rank of the first relevant document
    rank_first_relevant = round(sum([result['RFR'] for result in results]) / len(results), 4)
    print(f"Rank of the first relevant document = {rank_first_relevant}")
    # calculate the NDCG@10
    ndcg = round(sum([result['NDCG@10'] for result in results]) / len(results), 4)
    print(f"NDCG@10 = {ndcg}")

    with open(avg_results_file_path, "w") as f:
        f.write("MAP,MRR@10,RFR,NDCG@10\n")
        f.write(f"{mean_average_precision},{mrr},{rank_first_relevant},{ndcg}\n")


if __name__ == "__main__":
    test("./data/patents_to_publications.jsonl")