import os
import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class RagasEvaluator:
    def __init__(self, model_name: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        """
        Initializes the Ragas evaluator with necessary LLM and Embedding judge models.
        """
        # Ensure API key is set before proceeding
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
            
        # Initialize LangChain wrappers for the evaluator
        self.evaluator_llm = ChatOpenAI(model=model_name, temperature=0)
        self.evaluator_embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Bind the judges directly to the Ragas metric definitions
        faithfulness.llm = self.evaluator_llm
        answer_relevancy.llm = self.evaluator_llm
        answer_relevancy.embeddings = self.evaluator_embeddings
        
        # Track selected metrics
        self.metrics = [faithfulness, answer_relevancy]

    def compute_generation_metrics(
        self, 
        questions: List[str], 
        answers: List[str], 
        contexts: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Computes Faithfulness and Answer Relevancy for a batch of RAG generations.
        
        :param questions: List of user queries.
        :param answers: List of system generated responses.
        :param contexts: List of lists containing retrieved context strings used for each answer.
        :return: A dictionary containing overall summary metrics and a detailed pandas DataFrame.
        """
        # 1. Input Validation
        if not (len(questions) == len(answers) == len(contexts)):
            raise ValueError("All input lists (questions, answers, contexts) must have the exact same length.")
        
        # 2. Format data into the structured schema Ragas expects
        data_samples = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }
        dataset = Dataset.from_dict(data_samples)
        
        # 3. Execute the Ragas evaluation harness
        score_results = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )
        
        # 4. Format and return results
        return {
            "summary_scores": dict(score_results),
            "detailed_report": score_results.to_pandas()
        }

# ==========================================
# Example Execution Workflow
# ==========================================
"""if __name__ == "__main__":
    # Mock credentials configuration
    os.environ["OPENAI_API_KEY"] = "sk-..."

    # Define test data batches
    test_questions = [
        "What is the capital of France?",
        "How do plants generate food?"
    ]
    test_answers = [
        "The capital of France is Paris.",
        "Plants synthesize food via photosynthesis, turning sunlight into chemical energy."
    ]
    test_contexts = [
        ["Paris is the capital and most populous city of France."],
        ["Photosynthesis is a process used by plants to convert light energy into chemical energy."]
    ]

    try:
        # Instantiate the evaluation class
        evaluator = RagasEvaluator(model_name="gpt-4o-mini")
        
        # Run generation metrics function
        report = evaluator.compute_generation_metrics(
            questions=test_questions,
            answers=test_answers,
            contexts=test_contexts
        )
        
        # Display aggregated scores
        print("\n=== Aggregated Dataset Metrics ===")
        for metric, value in report["summary_scores"].items():
            print(f"{metric}: {value:.4f}")
            
        # Display sample breakdown
        print("\n=== Row-level Evaluation Breakdown ===")
        print(report["detailed_report"][["question", "faithfulness", "answer_relevancy"]])
        
    except ValueError as e:
        print(f"Initialization/Validation Error: {e}")
"""
