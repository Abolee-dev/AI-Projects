"""
hadoop_chatbot.py
===================

This module contains a collection of functions and classes that demonstrate
how to build a scalable data‑driven chatbot for IT service management
analytics.  The code shows how to connect to an on‑premise Hadoop data
warehouse via Hive or HDFS, collect data from Change Management, Incident
Management and configuration management (UCMDB) sources, generate
periodic insight reports (daily, weekly, monthly, quarterly and yearly),
and expose the data through a retrieval‑augmented LLM chatbot.  The
intention is to provide a clear template that can be expanded for
production deployments.

Key assumptions and prerequisites:

* The Hadoop cluster exposes a HiveServer2 endpoint.  According to the
  PyHive documentation, the simplest way to connect is to install
  `sasl`, `thrift`, `thrift‑sasl` and `pyhive` and then call
  `hive.Connection`【605795897163832†L151-L176】.  This is the approach used
  below.  For clusters secured with Kerberos, the `auth='KERBEROS'`
  option and a valid principal/keytab should be supplied.
* If direct HDFS access is required, `pyarrow` can be used.  The
  reference article shows how to configure environment variables, run
  `kinit` and instantiate `pyarrow.fs.HadoopFileSystem` to connect to
  a Kerberos‑secured namenode【4208015444759†L33-L64】.  This is included in
  the optional `connect_hdfs` function.
* The data warehouse contains three logical tables:
    - `change_management` – each row represents a change ticket.  It
      includes fields such as `change_id`, `server_name`, `service`,
      `change_status` (e.g. "Successful"/"Unsuccessful"), `mode`
      (e.g. "Mode‑2", "Manual"), `start_time`, `end_time` and
      `related_incident_id`.
    - `incident_management` – each row is an incident record with fields
      like `incident_id`, `server_name`, `service`, `severity` (e.g.
      "M1", "M2"), `disruptive` (boolean), `root_cause_change_id` and
      `opened_time`.
    - `ucmdb` – configuration management database; includes metadata
      about servers and services.  At minimum it contains the
      `server_name` and `service` columns so the data can be joined
      against the other tables.

The module is organised into the following sections:

1. **Connections** – helper functions to connect to Hive or HDFS.
2. **Data Extraction** – functions to fetch Change, Incident and UCMDB
   data for a given server and service.
3. **Data Pre‑processing and Aggregation** – functions to compute daily,
   weekly, monthly, quarterly and yearly metrics such as
   successful vs unsuccessful changes, Mode‑2 vs Manual changes,
   distribution of incidents by severity (M1/M2), disruptive incidents
   and the relationship between changes and incidents.
4. **Vector Store and LLM** – functions to build a vector store from
   aggregated metrics, instantiate an LLM (OpenAI or local) and create
   a Retrieval‑Augmented Generation (RAG) chain.  The RAG concept
   leverages large language models together with a custom knowledge
   base so that responses can reflect up‑to‑date, organisation‑specific
   data【351229365228803†L44-L59】.
5. **FastAPI Application** – a simple REST API exposing the chatbot via
   `/chat` endpoint.  This can be run locally or containerised and
   deployed on Kubernetes or other orchestration platforms.

Note: This example is intentionally simplified for clarity.  Real
deployments should include connection pooling, error handling,
authentication, caching, logging and monitoring.  Sensitive
information such as database credentials or API keys should be loaded
from environment variables or secret management systems rather than
hard‑coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
from pyhive import hive  # type: ignore

# Optional: import pyarrow if direct HDFS access is needed
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.fs  # type: ignore
except ImportError:
    pa = None  # pyarrow is optional

# Optional: import pyspark for large‑scale data processing
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, count as spark_count, sum as spark_sum
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

# LLM and RAG imports (LangChain)
try:
    from langchain.docstore.document import Document
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
except ImportError:
    # Fail gracefully if LangChain is not installed
    Document = None  # type: ignore
    Chroma = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    ChatOpenAI = None  # type: ignore
    RetrievalQA = None  # type: ignore


########################################
# Connection helpers
########################################

def connect_to_hive(
    host: str,
    port: int,
    username: str,
    password: Optional[str] = None,
    database: str = "default",
    auth: str = "NOSASL",
    kerberos_service_name: Optional[str] = None,
) -> hive.Connection:
    """Create a connection to Hive using PyHive.

    PyHive provides a DB‑API interface to HiveServer2.  According to the
    Edureka community answer, after installing `sasl`, `thrift`,
    `thrift‑sasl` and `PyHive`, you can connect to Hive with

    .. code-block:: python

        from pyhive import hive
        conn = hive.Connection(host="YOUR_HIVE_HOST", port=PORT, username="YOU")

    【605795897163832†L151-L176】.

    Parameters
    ----------
    host : str
        The hostname or IP address of the HiveServer2 service.
    port : int
        The port on which HiveServer2 listens (default is often 10000).
    username : str
        Username for authentication (ignored if Kerberos is used).
    password : Optional[str]
        Optional password (for password authentication modes).
    database : str
        Hive database (schema) to use.
    auth : str
        Authentication mechanism.  Options include "NONE", "NOSASL",
        "KERBEROS", etc.  For secure clusters with Kerberos, set
        auth="KERBEROS" and provide the service name.
    kerberos_service_name : Optional[str]
        Kerberos service principal for HiveServer2.

    Returns
    -------
    hive.Connection
        A connection object that can be used with `cursor()` or
        `pd.read_sql()`.
    """
    return hive.Connection(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        auth=auth,
        kerberos_service_name=kerberos_service_name,
    )


def connect_to_hdfs(
    host: str,
    port: int = 8020,
    kerb_ticket: Optional[str] = None,
    extra_conf: Optional[Dict[str, str]] = None,
) -> Optional[pa.fs.HadoopFileSystem]:
    """Connect to HDFS using PyArrow.

    The referenced article demonstrates how to set the necessary
    environment variables (HADOOP_HOME, HADOOP_CONF_DIR, ARROW_LIBHDFS_DIR,
    CLASSPATH), run `kinit` and instantiate a HadoopFileSystem object
    【4208015444759†L33-L64】.  For secure clusters, a Kerberos ticket cache is
    required (`kerb_ticket` points to the ticket file).

    Parameters
    ----------
    host : str
        Namenode host name.
    port : int
        Namenode port (default 8020 for HDFS).
    kerb_ticket : Optional[str]
        Path to the Kerberos ticket cache (e.g., `/tmp/krb5cc_user`).  If
        omitted, PyArrow will use the default ticket cache.
    extra_conf : Optional[Dict[str, str]]
        Additional configuration parameters (e.g., for HA setups).

    Returns
    -------
    pyarrow.fs.HadoopFileSystem or None
        A filesystem object if PyArrow is available, otherwise None.
    """
    if pa is None:
        raise RuntimeError("pyarrow is not installed.  Install with `pip install pyarrow`.")
    return pa.fs.HadoopFileSystem(host=host, port=port, kerb_ticket=kerb_ticket, extra_conf=extra_conf)


########################################
# Data extraction functions
########################################

def fetch_change_data(
    conn: hive.Connection,
    server_name: str,
    service_name: str,
    table_name: str = "change_management",
) -> pd.DataFrame:
    """Fetch change management data for a specific server and service.

    The query selects relevant columns needed for analytics.  You may need to
    adjust column names to match your schema.
    """
    query = f"""
        SELECT
            change_id,
            server_name,
            service,
            change_status,
            mode,
            start_time,
            end_time,
            related_incident_id
        FROM {table_name}
        WHERE server_name = '{server_name}'
          AND service = '{service_name}'
    """
    return pd.read_sql(query, conn)


def fetch_incident_data(
    conn: hive.Connection,
    server_name: str,
    service_name: str,
    table_name: str = "incident_management",
) -> pd.DataFrame:
    """Fetch incident management data for a specific server and service.

    Retrieves basic incident details including severity and whether the
    incident was disruptive.  Adjust column names to match your schema.
    """
    query = f"""
        SELECT
            incident_id,
            server_name,
            service,
            severity,
            disruptive,
            root_cause_change_id,
            opened_time
        FROM {table_name}
        WHERE server_name = '{server_name}'
          AND service = '{service_name}'
    """
    return pd.read_sql(query, conn)


def fetch_ucmdb_data(
    conn: hive.Connection,
    server_name: str,
    service_name: str,
    table_name: str = "ucmdb",
) -> pd.DataFrame:
    """Fetch UCMDB (configuration management) data for a server/service.

    This function returns metadata that can be merged with incident and
    change data.  Extend the selected columns based on your UCMDB schema.
    """
    query = f"""
        SELECT
            server_name,
            service,
            other_attributes
        FROM {table_name}
        WHERE server_name = '{server_name}'
          AND service = '{service_name}'
    """
    return pd.read_sql(query, conn)


########################################
# Data preparation and aggregation
########################################

def preprocess_change_data(change_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the change data.

    Converts timestamps to pandas datetime, derives success flag and normalises
    mode names.  Additional transformations can be added as needed.
    """
    df = change_df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    # Flag successful changes (case‑insensitive match on status)
    df["is_successful"] = df["change_status"].str.lower().str.contains("success")
    # Normalise mode names
    df["mode"] = df["mode"].str.strip().str.lower()
    df["mode_type"] = df["mode"].apply(lambda m: "mode‑2" if "mode" in m and "2" in m else ("manual" if "manual" in m else m))
    return df


def preprocess_incident_data(incident_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the incident data.

    Converts timestamps, standardises severity codes and ensures boolean type
    for the disruptive flag.
    """
    df = incident_df.copy()
    df["opened_time"] = pd.to_datetime(df["opened_time"])
    df["severity"] = df["severity"].str.strip().str.upper()
    # Convert disruptive indicator to boolean (assumes values like 1/0 or 'yes'/'no')
    df["disruptive"] = df["disruptive"].map(lambda x: str(x).lower() in ["1", "true", "yes", "y"])
    return df


def aggregate_periodic_metrics(
    change_df: pd.DataFrame,
    incident_df: pd.DataFrame,
    periods: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute periodic metrics for changes and incidents.

    Parameters
    ----------
    change_df : pd.DataFrame
        Preprocessed change management data.
    incident_df : pd.DataFrame
        Preprocessed incident management data.
    periods : Optional[List[str]]
        List of pandas offset aliases representing aggregation intervals.
        Defaults to daily ('D'), weekly ('W'), monthly ('M'), quarterly ('Q') and
        yearly ('A').  See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary mapping each period alias to a DataFrame of aggregated
        metrics.  Each DataFrame will include counts and proportions of
        successful vs unsuccessful changes, Mode‑2 vs Manual changes, incident
        severity distributions, disruptive incidents, and cross‑relationships.
    """
    if periods is None:
        periods = ["D", "W", "M", "Q", "A"]  # Daily, Weekly, Monthly, Quarterly, Annual

    metrics: Dict[str, pd.DataFrame] = {}
    for period in periods:
        # Aggregate change data
        change_group = change_df.groupby(pd.Grouper(key="start_time", freq=period))
        change_metrics = change_group.agg(
            total_changes=("change_id", "count"),
            successful_changes=("is_successful", "sum"),
        ).reset_index()
        change_metrics["success_rate"] = change_metrics["successful_changes"] / change_metrics["total_changes"].replace(0, pd.NA)

        # Mode distribution
        mode_counts = change_df.groupby([
            pd.Grouper(key="start_time", freq=period), "mode_type"
        ]).size().unstack(fill_value=0)
        for col in mode_counts.columns:
            change_metrics[f"mode_{col}_count"] = mode_counts[col].values

        # Aggregate incident data
        incident_group = incident_df.groupby(pd.Grouper(key="opened_time", freq=period))
        incident_metrics = incident_group.agg(
            total_incidents=("incident_id", "count"),
            disruptive_incidents=("disruptive", "sum"),
        ).reset_index()
        # Severity distribution
        severity_counts = incident_df.groupby([
            pd.Grouper(key="opened_time", freq=period), "severity"
        ]).size().unstack(fill_value=0)
        for col in severity_counts.columns:
            incident_metrics[f"severity_{col}_count"] = severity_counts[col].values

        # Merge change and incident metrics on the period index
        period_metrics = pd.merge(
            change_metrics,
            incident_metrics,
            left_on="start_time",
            right_on="opened_time",
            how="outer",
        ).rename(columns={"start_time": "period"})
        period_metrics = period_metrics.drop(columns=["opened_time"])  # clean up

        # Calculate relationships: changes causing incidents and incidents caused by changes
        # For each period, count number of change tickets that have related incidents and vice versa
        # We'll compute this by joining on foreign keys.
        # For the given period, filter the data frames
        for _, group_data in change_group:
            pass  # placeholder to ensure loop executes at least once

        def relation_counts(start, end):
            changes_period = change_df[(change_df["start_time"] >= start) & (change_df["start_time"] < end)]
            incidents_period = incident_df[(incident_df["opened_time"] >= start) & (incident_df["opened_time"] < end)]
            # Count changes that reference incidents
            changes_with_inc = changes_period["related_incident_id"].notna().sum()
            # Count incidents that reference changes
            incidents_with_change = incidents_period["root_cause_change_id"].notna().sum()
            return changes_with_inc, incidents_with_change

        # Compute start and end boundaries for each row
        boundaries: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for i, current_period in enumerate(period_metrics["period"]):
            if pd.isna(current_period):
                boundaries.append((pd.Timestamp.min, pd.Timestamp.min))
            else:
                if period == "D":
                    start = current_period
                    end = current_period + pd.Timedelta(days=1)
                elif period == "W":
                    start = current_period - pd.Timedelta(days=current_period.weekday())
                    end = start + pd.Timedelta(weeks=1)
                elif period == "M":
                    start = current_period.replace(day=1)
                    end = (start + pd.offsets.MonthEnd()).normalize() + pd.Timedelta(days=1)
                elif period == "Q":
                    month = ((current_period.month - 1) // 3) * 3 + 1
                    start = current_period.replace(month=month, day=1)
                    end = (start + pd.offsets.QuarterEnd()).normalize() + pd.Timedelta(days=1)
                elif period == "A":
                    start = current_period.replace(month=1, day=1)
                    end = (start + pd.offsets.YearEnd()).normalize() + pd.Timedelta(days=1)
                else:
                    # fallback to daily
                    start = current_period
                    end = start + pd.Timedelta(days=1)
                boundaries.append((start, end))

        rel_change_counts: List[int] = []
        rel_incident_counts: List[int] = []
        for start, end in boundaries:
            if start == pd.Timestamp.min:
                rel_change_counts.append(0)
                rel_incident_counts.append(0)
            else:
                c, i = relation_counts(start, end)
                rel_change_counts.append(c)
                rel_incident_counts.append(i)

        period_metrics["changes_with_incidents"] = rel_change_counts
        period_metrics["incidents_with_changes"] = rel_incident_counts

        metrics[period] = period_metrics
    return metrics


########################################
# Vector store and retrieval‑augmented LLM
########################################

def build_vector_store_from_metrics(
    metrics: Dict[str, pd.DataFrame],
    persist_directory: str = ".vectordb",
    embedding_model_name: str = "text-embedding-ada-002",
    use_local_embeddings: bool = False,
) -> Chroma:
    """Create a Chroma vector store from aggregated metrics.

    Each period’s DataFrame is transformed into a `Document` whose content
    consists of a human‑readable summary of the metrics.  The documents
    are embedded using an OpenAI embedding model by default.  If
    `use_local_embeddings` is True, LangChain will attempt to load
    a local embedding model (e.g. via `SentenceTransformers`), which can be
    useful when privacy or cost concerns favour local models.  The
    resulting vector store is persisted on disk, enabling reuse across
    sessions.

    Returns
    -------
    Chroma
        A vector store ready to be used in a retrieval chain.
    """
    if Document is None or Chroma is None:
        raise RuntimeError(
            "LangChain is not installed.  Install with `pip install langchain chromadb openai` to use this feature."
        )

    docs: List[Document] = []
    # Create documents from metrics
    for period_alias, df in metrics.items():
        # Convert DataFrame to a markdown table string for readability
        table_md = df.to_markdown(index=False)
        # Compose a descriptive header
        header = f"Periodic metrics for period '{period_alias}':\n"
        content = header + table_md
        docs.append(Document(page_content=content, metadata={"period": period_alias}))

    # Choose embedding model
    if use_local_embeddings:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        embeddings = HuggingFaceEmbeddings()
    else:
        embeddings = OpenAIEmbeddings(model=embedding_model_name)

    # Create or load vector store
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


def create_retrieval_qa_chain(
    vectordb: Chroma,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    use_local_llm: bool = False,
) -> RetrievalQA:
    """Build a RetrievalQA chain using a vector store and an LLM.

    If `use_local_llm` is True, the function will attempt to use a local
    LLM served via the `Ollama` interface, which exposes an OpenAI‑compatible
    API endpoint.  Otherwise, it uses OpenAI’s hosted API.  For local
    usage you must set `OPENAI_API_BASE` to the base URL of your local
    model (e.g. http://localhost:11434/v1).

    Returns
    -------
    RetrievalQA
        A chain that answers queries using retrieval‑augmented generation.
    """
    if RetrievalQA is None:
        raise RuntimeError(
            "LangChain is not installed.  Install with `pip install langchain openai` to use this feature."
        )
    # Set up LLM
    if use_local_llm:
        # When running a local LLM via Ollama, set OPENAI_API_KEY to a dummy value
        # and OPENAI_API_BASE to the endpoint provided by Ollama (e.g. http://localhost:11434/v1).
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    else:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa_chain


########################################
# FastAPI application
########################################

def create_app(qa_chain: RetrievalQA):
    """Create a FastAPI application exposing a /chat endpoint."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="IT Service Analytics Chatbot")

    class QueryRequest(BaseModel):
        query: str

    @app.post("/chat")
    async def chat(request: QueryRequest):
        response = qa_chain.run(request.query)
        return {"response": response}

    return app


########################################
# Main pipeline (example usage)
########################################

def build_and_run_chatbot(
    hive_host: str,
    hive_port: int,
    hive_user: str,
    server_name: str,
    service_name: str,
    persist_directory: str = ".vectordb",
    use_local_llm: bool = False,
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = "gpt-3.5-turbo",
):
    """End‑to‑end pipeline to build and run the chatbot.

    1. Connect to Hive and fetch change, incident and UCMDB data for the
       specified server/service.
    2. Preprocess and aggregate the data into periodic metrics.
    3. Build a vector store from the metrics and create a retrieval QA chain.
    4. Launch a FastAPI application that uses the QA chain to answer
       questions.

    This function is meant for demonstration; adapt it for production
    deployment.
    """
    # Step 1: Connect to Hive
    conn = connect_to_hive(host=hive_host, port=hive_port, username=hive_user)
    # Step 2: Fetch data
    change_df = fetch_change_data(conn, server_name, service_name)
    incident_df = fetch_incident_data(conn, server_name, service_name)
    ucmdb_df = fetch_ucmdb_data(conn, server_name, service_name)
    # Step 3: Preprocess
    change_df_prep = preprocess_change_data(change_df)
    incident_df_prep = preprocess_incident_data(incident_df)
    # Step 4: Aggregate metrics
    metrics = aggregate_periodic_metrics(change_df_prep, incident_df_prep)
    # Step 5: Build vector store
    vectordb = build_vector_store_from_metrics(
        metrics,
        persist_directory=persist_directory,
        embedding_model_name=embedding_model,
        use_local_embeddings=use_local_llm,
    )
    # Step 6: Create retrieval QA chain
    qa_chain = create_retrieval_qa_chain(
        vectordb,
        model_name=llm_model,
        temperature=0.0,
        use_local_llm=use_local_llm,
    )
    # Step 7: Build API
    app = create_app(qa_chain)
    return app


if __name__ == "__main__":
    # Example usage.  The following values should be replaced with real
    # connection details and executed from the command line.  When run
    # directly, this script starts a FastAPI application served via
    # Uvicorn.  In a Kubernetes deployment, Uvicorn/Gunicorn workers can
    # be containerised and orchestrated.
    import uvicorn

    HIVE_HOST = os.environ.get("HIVE_HOST", "localhost")
    HIVE_PORT = int(os.environ.get("HIVE_PORT", "10000"))
    HIVE_USER = os.environ.get("HIVE_USER", "hadoop")
    SERVER_NAME = os.environ.get("IT_SERVER", "gb-gf")
    SERVICE_NAME = os.environ.get("IT_SERVICE", "itservice")

    # Build the chatbot app
    chatbot_app = build_and_run_chatbot(
        hive_host=HIVE_HOST,
        hive_port=HIVE_PORT,
        hive_user=HIVE_USER,
        server_name=SERVER_NAME,
        service_name=SERVICE_NAME,
        persist_directory="./vectordb",
        use_local_llm=bool(os.environ.get("USE_LOCAL_LLM")),
        embedding_model=os.environ.get("EMBED_MODEL", "text-embedding-ada-002"),
        llm_model=os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
    )
    # Run the API
    uvicorn.run(chatbot_app, host="0.0.0.0", port=8000)