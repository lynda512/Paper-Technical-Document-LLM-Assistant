from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMClient:
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.max_context = 512  # keep small to be safe

    def chat(self, system: str, messages: List[dict]) -> str:
        text = system + "\n"
        for m in messages:
            text += f"{m.get('role','user')}: {m['content']}\n"
        text += "assistant:"

        # Tokenize then truncate to last N tokens
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"][:, -self.max_context :]
        attention_mask = enc["attention_mask"][:, -self.max_context :]
        inputs = {"input_ids": input_ids.to(self.device),
                  "attention_mask": attention_mask.to(self.device)}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
