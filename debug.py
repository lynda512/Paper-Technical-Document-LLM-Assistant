# debug_app_start.py
import traceback

print("Importing app ...")
try:
    from src.ui.app import app  # only import, don't run uvicorn
    print("OK: app imported, FastAPI instance =", app)
except Exception:
    print("ERROR while importing app:\n")
    traceback.print_exc()
