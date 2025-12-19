import sys
import importlib
import traceback

# Add *src* to path
sys.path.insert(0, r"C:\Users\lynda\Documents\AI Eng\projects\paper-llm-assistant\src")

pkgs = [
    "embeddings",
    "ingestion",
    "llm",
    "preprocessing",
    "rag_pipeline",
    "retrieval",
    "ui",
    "utils",
]

failed = []
for p in pkgs:
    try:
        importlib.import_module(p)
        print("OK", p)
    except Exception as e:
        print("FAILED", p, e)
        traceback.print_exc()
        failed.append(p)

if failed:
    print("\nFAILED IMPORTS:", failed)
    sys.exit(1)
else:
    print("\nALL IMPORTS OK")
    sys.exit(0)
