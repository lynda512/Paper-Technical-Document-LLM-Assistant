from src.llm.client import LLMClient

print("Loading test model...")
llm = LLMClient("sshleifer/tiny-gpt2")
print("Loaded OK.")

ans = llm.chat("You are a test model.", [{"role": "user", "content": "Say hello in one sentence."}])
print("Answer:", ans)
