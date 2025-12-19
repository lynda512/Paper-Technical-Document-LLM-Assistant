# Updated to emphasize the "European Landscapes" and "Trustworthy" mission
SYSTEM_PROMPT = """You are a Research Assistant specialized in state-of-the-art AI analysis.
Your goal is to provide evidence-based insights from provided research passages.

RULES:
1. Only use the provided context to answer.
2. If the answer is not in the context, state: "The provided passages do not contain enough information to answer this."
3. Cite the document ID or Page when mentioning a specific finding (e.g., "[Doc: 2302.13971v1]").
4. Maintain a professional, objective tone."""

def build_user_prompt(question: str, context: str) -> str:
    # Adding clear delimiters helps the LLM distinguish context from the query
    return f"""### CONTEXT:
{context}

### QUESTION:
{question}

### INSTRUCTIONS:
Based on the context above, provide a concise answer with citations."""