"""Evaluate candidate rankings with metrics owned by this repository."""

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers.util import cos_sim
from sklearn.metrics import average_precision_score, ndcg_score


logger = logging.getLogger(__name__)
METRICS = ("MAP", "MRR@10", "RFR", "NDCG@10")


def score_ranking(relevance, scores):
    """Calculate all ranking metrics for one query."""
    score_tensor = torch.as_tensor(scores)
    ranking = torch.argsort(-score_tensor).detach().cpu().numpy()
    score_values = score_tensor.detach().cpu().numpy()
    first_relevant_rank = next(
        rank
        for rank, candidate_index in enumerate(ranking, start=1)
        if relevance[candidate_index]
    )

    return {
        "MAP": float(average_precision_score(relevance, score_values)),
        "MRR@10": (
            1.0 / first_relevant_rank
            if first_relevant_rank <= 10
            else 0.0
        ),
        "RFR": first_relevant_rank,
        "NDCG@10": float(ndcg_score([relevance], [score_values], k=10)),
    }


def write_results(results, output_path, model_name):
    """Write aggregate and per-query metrics using the existing filenames."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    averages = {
        metric: sum(result[metric] for result in results) / len(results)
        for metric in METRICS
    }

    summary_path = output_path / f"RerankingEvaluator_{model_name}_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["epoch", "steps"] + list(METRICS))
        metric_values = [averages[metric] for metric in METRICS]
        writer.writerow([-1, -1] + metric_values)

    sample_path = output_path / f"RerankingEvaluator_{model_name}_whole_sample_results.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["epoch", "sample_nr"] + list(METRICS))
        for sample_number, result in enumerate(results, start=1):
            metric_values = [result[metric] for metric in METRICS]
            writer.writerow([-1, sample_number] + metric_values)

    return averages


def evaluate_ranking(model, samples, output_path, model_name, batch_size=64, show_progress_bar=True):
    """Embed once, score every candidate list, and write all metrics."""
    samples = [
        sample
        for sample in samples
        if sample["positive"] and sample["negative"]
    ]
    if not samples:
        raise ValueError("No queries have both positive and negative candidates")

    all_queries = [sample["query"] for sample in samples]
    query_embeddings = model.encode(
        all_queries,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    candidates_by_query = [
        sample["positive"] + sample["negative"]
        for sample in samples
    ]

    all_candidates = []
    for candidates in candidates_by_query:
        all_candidates.extend(candidates)

    all_candidate_embeddings = model.encode(
        all_candidates,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    candidate_embeddings_by_query = torch.split(
        all_candidate_embeddings,
        [len(candidates) for candidates in candidates_by_query],
    )

    results = []
    for query_embedding, sample, candidate_embeddings in zip(
        query_embeddings,
        samples,
        candidate_embeddings_by_query,
    ):
        scores = cos_sim(query_embedding, candidate_embeddings)[0]
        relevance = np.array(
            [1] * len(sample["positive"])
            + [0] * len(sample["negative"])
        )
        results.append(score_ranking(relevance, scores))

    averages = write_results(results, output_path, model_name)
    logger.info(
        "MAP: %.4f; MRR@10: %.4f; RFR: %.2f; NDCG@10: %.4f",
        averages["MAP"],
        averages["MRR@10"],
        averages["RFR"],
        averages["NDCG@10"],
    )
    return averages
