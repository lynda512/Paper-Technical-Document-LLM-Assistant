import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMClient:
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        # Detect GPU for "Deep Learning" hands-on experience
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use fast tokenizers for industrial applicability
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.max_context = 2048 

    def chat(self, system: str, messages: List[Dict]) -> str:
        """
        Uses Chat Templates to ensure the model follows instructions.
        Crucial for 'Trustworthy AI' applications.
        """
        # Formulate a proper chat structure
        formatted_messages = [{"role": "system", "content": system}] + messages
        
        # Applying a chat template is the standard for modern LLMs
        input_ids = self.tokenizer.apply_chat_template(
            formatted_messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.1, # Low temperature for "faithfulness" to research text
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (the assistant's response)
        response = self.tokenizer.decode(
            output[0][len(input_ids[0]):], 
            skip_special_tokens=True
        )
        return response.strip()