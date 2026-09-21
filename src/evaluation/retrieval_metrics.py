import numpy as np

def precision_at_k(actual, predicted, k):
    """
    Calculates Precision@K for a single user/query.
    actual: set or list of ground truth items
    predicted: ordered list of recommended items
    """
    if not actual:
        return 0.0
    
    # Take the top K predictions
    top_k_pred = predicted[:k]
    
    # Calculate how many of the top K predictions are relevant
    hits = len(set(top_k_pred) & set(actual))
    
    return hits / k

def recall_at_k(actual, predicted, k):
    """
    Calculates Recall@K for a single user/query.
    actual: set or list of ground truth items
    predicted: ordered list of recommended items
    """
    if not actual:
        return 0.0
    
    # Take the top K predictions
    top_k_pred = predicted[:k]
    
    # Calculate how many of the top K predictions are relevant
    hits = len(set(top_k_pred) & set(actual))
    
    return hits / len(actual)

def rr_at_k(actual, predicted, k):
    """
    Calculates Reciprocal Rank (RR)@K for a single user/query.
    Returns 1 / rank of the first relevant item, or 0 if none found in top K.
    """
    if not actual:
        return 0.0
    
    # Search for the first relevant item within the top K predictions
    for rank, item in enumerate(predicted[:k], start=1):
        if item in actual:
            return 1.0 / rank
            
    return 0.0

def evaluate_metrics(actual_batch, predicted_batch, k):
    """
    Evaluates Mean Precision@K, Mean Recall@K, and MRR@K across a batch.
    """
    p_scores = []
    r_scores = []
    rr_scores = []
    
    for actual, predicted in zip(actual_batch, predicted_batch):
        p_scores.append(precision_at_k(actual, predicted, k))
        r_scores.append(recall_at_k(actual, predicted, k))
        rr_scores.append(rr_at_k(actual, predicted, k))
        
    return {
        f"Mean Precision@{k}": np.mean(p_scores),
        f"Mean Recall@{k}": np.mean(r_scores),
        f"MRR@{k}": np.mean(rr_scores)
    }

# ==========================================
# Example Usage
# ==========================================
"""if __name__ == "__main__":
    # Ground truth relevant items for 2 users
    actual_data = [
        [10, 20, 30],       # User 1 likes items 10, 20, 30
        [40, 50]            # User 2 likes items 40, 50
    ]
    
    # Top-ranked recommendations given to those 2 users
    predicted_data = [
        [99, 10, 88, 30, 77], # User 1 got (Item 10 at rank 2, Item 30 at rank 4)
        [40, 99, 50, 88, 77]  # User 2 got (Item 40 at rank 1, Item 50 at rank 3)
    ]
    
    K = 3
    results = evaluate_metrics(actual_data, predicted_data, k=K)
    
    print(f"--- Evaluation at K = {K} ---")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
"""
