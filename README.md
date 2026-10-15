
# RAG Document Intelligence System 

A Retrieval-Augmented Generation (RAG) application that allows users to upload documents, generate embeddings, and manage vector data for document-based question answering.

## Prerequisites
Make sure you have installed the following:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Groq API Key](https://console.groq.com/keys)

## Installation and Setup
### Step 1: Configure Environment Variables

Copy the example environment file:
```

cp .env.example .env.docker

```

Open `.env.docker` and configure your Groq API key:
```

GROQ_API_KEY=your_groq_api_key

```

Replace `your_groq_api_key` with your actual Groq API key.

### 2. Start the Application

Run the following command to start the application and its associated services in detached mode:
```

docker compose up -d

```

### 3. Download the Embedding Model

Pull the `nomic-embed-text:v1.5` embedding model into Ollama:
```

docker exec -it ollama ollama pull nomic-embed-text:v1.5

```

This model is used to generate embeddings for document processing and semantic search.

### 4. Verify the Installed Model

To verify that the embedding model has been downloaded successfully, run:
```

docker exec -it ollama ollama list

```

Ensure that `nomic-embed-text:v1.5` appears in the list of installed models.

### 5. Access the Application

Once all services are running, open the following URL in your browser:

**http://127.0.0.0:8080**

You can now start uploading documents and managing your RAG data.

## Manage RAG Data

The application provides an interface for uploading documents and managing previously processed RAG data.

### Upload Documents

1. Click the **Show Upload** button.
2. Select one or more documents from your device.
3. Upload the selected files to add their content to the RAG system.

**Supported file formats:**
- `.txt` — Plain text files
- `.pdf` — PDF documents

Multiple files can be uploaded at once.

### Clear RAG Data

Click the **Clear Data** button to remove previously processed RAG data, including the associated vector data.

> **Note:** Clearing RAG data removes previously stored information used for document retrieval. Upload your documents again if you want to restore that data.

## Technology Stack

- **Docker & Docker Compose** — Containerization and service orchestration
- **Ollama** — Local embedding model management
- **Nomic Embed Text v1.5** — Text embedding generation
- **Groq API** — Large language model inference
- **RAG** — Retrieval-Augmented Generation for document-based question answering

## Troubleshooting

### Ollama Container Not Found

If the `ollama` container is not running, check the status of your Docker services:
```

docker compose ps

```

If necessary, restart the services:
```

docker compose up -d

```

### Embedding Model Not Found

Check the installed Ollama models:
```

docker exec -it ollama ollama list

```

If `nomic-embed-text:v1.5` is missing, download it again:
```

docker exec -it ollama ollama pull nomic-embed-text:v1.5

```

### Application Not Accessible

Verify that the containers are running:
```

docker compose ps

```

Check the application logs for errors:
```

docker compose logs -f <service_name>

```

### Updating Your Project
Whenever you make changes to the project and need to rebuild and restart the Docker containers, run the following command:
```

docker compose up -d --build
```

### Groq API Issues

Verify that your `GROQ_API_KEY` is correctly configured in `.env.docker` and that the key is valid.

If you change environment variables after starting the containers, recreate the services:
```

docker compose up -d --force-recreate

```




**Important:** Your setup assumes that Docker Compose is configured to load `.env.docker`. Docker Compose does not automatically use `.env.docker` as its default environment file. If your `compose.yaml` does not explicitly load it, start the application with:

```
docker compose --env-file .env.docker up -d
```