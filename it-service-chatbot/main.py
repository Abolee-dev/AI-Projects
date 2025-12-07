"""
Entry point for the IT service analytics chatbot.

This script imports the `build_and_run_chatbot` function from
`hadoop_chatbot` and uses environment variables to configure the
Hive connection and chatbot settings.  When run directly, it starts
a Uvicorn server exposing the FastAPI application on port 8000.

Environment variables:
  HIVE_HOST      – hostname of the HiveServer2 service
  HIVE_PORT      – port of HiveServer2 (default 10000)
  HIVE_USER      – Hive username
  IT_SERVER      – server name to filter data (e.g. "gb-gf")
  IT_SERVICE     – service name to filter data (e.g. "itservice")
  USE_LOCAL_LLM  – set to "true" to use a local LLM via Ollama
  EMBED_MODEL    – embedding model name (default "text-embedding-ada-002")
  LLM_MODEL      – LLM model name (default "gpt-3.5-turbo")

The module must be run from within a Docker container or on a host
where dependencies such as pyhive, pandas, pyarrow, pyspark,
langchain, chromadb, openai, fastapi and uvicorn are installed.
"""

import os
import uvicorn
from hadoop_chatbot import build_and_run_chatbot


# Build the chatbot application using environment variables
app = build_and_run_chatbot(
    hive_host=os.getenv("HIVE_HOST", "localhost"),
    hive_port=int(os.getenv("HIVE_PORT", "10000")),
    hive_user=os.getenv("HIVE_USER", "hadoop"),
    server_name=os.getenv("IT_SERVER", "gb-gf"),
    service_name=os.getenv("IT_SERVICE", "itservice"),
    persist_directory="./vectordb",
    use_local_llm=os.getenv("USE_LOCAL_LLM", "false").lower() == "true",
    embedding_model=os.getenv("EMBED_MODEL", "text-embedding-ada-002"),
    llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
)


if __name__ == "__main__":
    # Start the Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)