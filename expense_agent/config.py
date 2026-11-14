import os

from dotenv import load_dotenv

load_dotenv()

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
TASK_QUEUE = os.environ.get("TASK_QUEUE", "expense-task-queue")

OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
UI_STREAM_URL = os.environ.get("UI_STREAM_URL", "")

HISTORY_LENGTH_THRESHOLD = int(os.environ.get("HISTORY_LENGTH_THRESHOLD", "20"))
CONTEXT_STORE_PATH = os.environ.get("CONTEXT_STORE_PATH", "context_store.jsonl")
