from typing import List, Dict

def simple_qa_log(question: str, answer: str, metadata: Dict) -> Dict:
    return {
        "question": question,
        "answer": answer,
        "source_docs": metadata.get("source_docs", []),
        "latency_ms": metadata.get("latency_ms"),
        "notes": metadata.get("notes", ""),
    }

def qualitative_feedback_template() -> List[str]:
    return [
        "Was the main contribution correctly summarized?",
        "Did the answer reference relevant parts of the paper?",
        "Were limitations of the method mentioned?",
    ]
