# IT Service Analytics Chatbot

This project contains a reproducible code base to build and deploy a
retrieval‑augmented chatbot that surfaces insights from IT service
management data stored in an on‑prem Hadoop warehouse.  The chatbot
connects to Hive to collect change, incident and UCMDB data, produces
daily/weekly/monthly/quarterly/yearly metrics, stores these metrics in
a vector database and answers natural‑language questions via a FastAPI
endpoint.

## Directory Structure

```
it-service-chatbot/
├── Dockerfile           # Instructions to build the application container
├── README.md            # This documentation file
├── main.py              # Entry point that configures and launches the FastAPI app
├── hadoop_chatbot.py    # Core logic: data extraction, aggregation and chatbot
├── requirements.txt     # Python dependencies
└── k8s-deployment.yaml  # Example Kubernetes Deployment and Service manifest
```

## Getting Started Locally

1. **Clone or copy this directory** into your project repository.
2. Install the required Python packages (preferably in a virtualenv):

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Set the necessary environment variables for your Hive
   connection.  At a minimum you need `HIVE_HOST`, `HIVE_PORT` and
   `HIVE_USER`.  You can also override `IT_SERVER`, `IT_SERVICE`,
   `USE_LOCAL_LLM`, `EMBED_MODEL` and `LLM_MODEL` to adjust the
   chatbot’s behaviour.  For example:

   ```bash
   export HIVE_HOST=my-hive-host
   export HIVE_PORT=10000
   export HIVE_USER=hadoop
   export IT_SERVER=gb-gf
   export IT_SERVICE=itservice
   python main.py
   ```

4. Open your browser to `http://localhost:8000/docs` to explore the
   FastAPI auto‑generated documentation and test the `/chat` endpoint.

## Building the Docker Image

Run the following command from within the `it-service-chatbot` directory:

```bash
docker build -t it-service-chatbot:latest .
```

Then start a container, passing in the necessary environment variables:

```bash
docker run --rm -p 8000:8000 \
  -e HIVE_HOST=my-hive-host \
  -e HIVE_PORT=10000 \
  -e HIVE_USER=hadoop \
  -e IT_SERVER=gb-gf \
  -e IT_SERVICE=itservice \
  it-service-chatbot:latest
```

The service will be available at `http://localhost:8000/chat`.

## Kubernetes Deployment

The file `k8s-deployment.yaml` contains a basic Kubernetes
Deployment and Service.  Update the `image` field with your built
container image (pushed to a registry) and adjust environment
variables as needed.  Then deploy with:

```bash
kubectl apply -f k8s-deployment.yaml
```

This will create a deployment named `it-service-chatbot` and a
ClusterIP service exposing port 80 that forwards to the container’s
port 8000.  You can expose the service externally using an
Ingress or a LoadBalancer depending on your cluster setup.

## Notes

- The project uses **PyHive** to connect to Hive【605795897163832†L151-L176】.  Ensure the
  necessary dependencies (`sasl`, `thrift`, `thrift‑sasl`) are
  available in the container image.  If your Hadoop cluster uses
  Kerberos, set `auth='KERBEROS'` and configure `kerberos_service_name`.
- Direct HDFS access (via PyArrow) is optional; see the comments in
  `hadoop_chatbot.py` for details【4208015444759†L33-L64】.
- The chatbot uses a retrieval‑augmented generation (RAG) pipeline so
  that answers are grounded in your own data rather than the model’s
  training data【351229365228803†L44-L59】.  You can switch to a local LLM (via
  Ollama) by setting the `USE_LOCAL_LLM` environment variable to
  `true` and configuring `OPENAI_API_BASE` accordingly.

## Contributing

Feel free to extend the schema, metrics and UI to better fit your
organisation’s needs.  Pull requests and suggestions are welcome!