from transformers import AutoToknizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
  def __init__(self, model_name="Alibaba-NLP/gte-reranker-modernbert-base"):
    self.tokenizer= AutoTokenizer.from_pretrained(model_name)
    self.model=AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")
    self.model.eval
  def rerank(slef, query:str, condidates:list[dict], top_k : int=4) ->list[dict]:
        """
        candidates: list of dicts with at least a 'text' key
        (your fused chunks from RRF_Fusion.py)
        Returns candidates sorted by reranker score, trimmed to top_k.
        """
        pairs = [[query, c["text"]] for c in candidates]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            ).to(self.model.device)
            scores = self.model(**inputs).logits.squeeze(-1)

        for c, score in zip(candidates, scores.tolist()):
            c["rerank_score"] = score

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
