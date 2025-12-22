import openai
from typing import List, Dict

class LLMClient:
    def __init__(self, model_name: str = "local-model"):
        # mandatory /v1 suffix for LM Studio
        self.api_base = "http://127.0.0.1:1234/v1"
        self.api_key = "lm-studio" 
        
        self.client = openai.OpenAI(
            base_url=self.api_base, 
            api_key=self.api_key,
            timeout=120.0 
        )

    def chat(self, system: str, messages: List[Dict]) -> Dict:
        """
        Sends RAG context to Llama 3. 
        Always returns a dictionary to prevent 'NoneType' errors in UI.
        """
        formatted_messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = self.client.chat.completions.create(
                model="meta-llama-3-8b-instruct", 
                messages=formatted_messages,
                temperature=0.1, # Critical for 'Trustworthy AI' faithfulness
                max_tokens=900
            )
            return {"answer": response.choices[0].message.content, "status": "success"}
        except Exception as e:
            # Return a dict instead of a raw string to keep UI subscriptable
            return {
                "answer": f"❌ Connection Error: Ensure LM Studio server is started on port 1234. Detail: {str(e)}",
                "status": "error"
            }