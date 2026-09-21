"""
run_eval.py - sweep chunk size x retriever, save one results table.

Usage:
    python run_eval.py --testset testset.json --k 5
    python run_eval.py --testset testset.json --k 5 --with-generation   # adds Ragas (costs API calls)

testset.json format (one entry per question):
[
  {
    "question": "What optimizer does the paper use?",
    "ground_truth": "AdamW with lr 3e-4",            # optional, for your own checks
    "sources": [{"doc": "paper_a.pdf", "pages": [4, 5]}]
  }
]
Ground truth is labelled by document + page, NOT chunk id, so it stays valid
when the chunk size changes.
"""
import argparse
import json

import numpy as np
import pandas as pd

# Adjust these two imports to your file names.
from retrieval_metrics import precision_at_k, recall_at_k, rr_at_k
from ragas_evaluator import RagasEvaluator

CHUNK_SIZES = [256, 512, 1024]
RETRIEVERS = ["bm25", "dense", "hybrid"]


# ---------------------------------------------------------------------------
# HOOKS - connect these to your existing RAG project
# ---------------------------------------------------------------------------
def build_pipeline(chunk_size: int, retriever_type: str):
    """
    Re-index your documents with `chunk_size` and return an object with:
        .retrieve(question, k) -> list of {"text": str, "doc": str, "pages": [int, ...]}
        .answer(question, chunks) -> str
    Replace this with calls into your LangChain / FAISS code.
    """
    raise NotImplementedError("Connect build_pipeline() to your RAG code.")


# ---------------------------------------------------------------------------
# Ground-truth matching by document + page
# ---------------------------------------------------------------------------
def gt_keys(sources):
    return {f"{s['doc']}:{p}" for s in sources for p in s["pages"]}


def chunk_to_item(chunk, gt):
    """Map a retrieved chunk to a gt key if it overlaps one, else to its own key."""
    keys = [f"{chunk['doc']}:{p}" for p in chunk["pages"]]
    for key in keys:
        if key in gt:
            return key
    return keys[0] if keys else "unknown"


def run_config(testset, chunk_size, retriever_type, k, with_generation, evaluator):
    pipe = build_pipeline(chunk_size, retriever_type)
    p, r, rr = [], [], []
    questions, answers, contexts = [], [], []

    for ex in testset:
        gt = gt_keys(ex["sources"])
        if not gt:  # skip queries without ground truth instead of scoring them 0
            continue
        chunks = pipe.retrieve(ex["question"], k)
        predicted = [chunk_to_item(c, gt) for c in chunks]
        p.append(precision_at_k(gt, predicted, k))
        r.append(recall_at_k(gt, predicted, k))
        rr.append(rr_at_k(gt, predicted, k))

        if with_generation:
            questions.append(ex["question"])
            answers.append(pipe.answer(ex["question"], chunks))
            contexts.append([c["text"] for c in chunks])

    row = {
        "chunk_size": chunk_size,
        "retriever": retriever_type,
        f"precision@{k}": np.mean(p),
        f"recall@{k}": np.mean(r),
        f"mrr@{k}": np.mean(rr),
        "n_questions": len(p),
    }
    if with_generation:
        df = evaluator.compute_generation_metrics(questions, answers, contexts)["detailed_report"]
        row["faithfulness"] = df["faithfulness"].mean()
        row["answer_relevancy"] = df["answer_relevancy"].mean()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--with-generation", action="store_true")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    with open(args.testset, encoding="utf-8") as f:
        testset = json.load(f)

    evaluator = RagasEvaluator() if args.with_generation else None

    rows = []
    for cs in CHUNK_SIZES:
        for rt in RETRIEVERS:
            print(f"Running chunk_size={cs}, retriever={rt} ...")
            rows.append(run_config(testset, cs, rt, args.k, args.with_generation, evaluator))

    results = pd.DataFrame(rows).round(3)
    results.to_csv(args.out, index=False)
    print("\n", results.to_string(index=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
