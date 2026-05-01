# Aesthetic Diagnosis — RAG Search Engine

Intelligent search engine over aesthetic medicine and skincare training materials. Allows natural language queries and returns answers with references to the original sources.

## Architecture

```
Documents (PPTX, PDF, DOCX, XLSX, MP4)
    ↓ add_sources.py
Pinecone (vector database)
    ↓ search.py
Web UI (Streamlit) → GPT-4o-mini → Answer with sources
```

## Requirements

- Python 3.12
- Pipenv
- [Pinecone](https://www.pinecone.io/) account with an index created (`llama-text-embed-v2`)
- [OpenAI](https://platform.openai.com/) account

## Configuration

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_HOST=https://your-index.svc.pinecone.io
OPENAI_API_KEY=your_openai_api_key
```

Create a `credentials.yaml` file with the users allowed to access the app:

```yaml
credentials:
  usernames:
    username:
      email: email@example.com
      name: Display Name
      password: <bcrypt_hash>

cookie:
  expiry_days: 7
  key: a_long_secret_key
  name: aesthetic_diagnosis_auth
```

To generate a password hash:

```bash
pipenv run python -c "import streamlit_authenticator as stauth; print(stauth.Hasher().hash('your_password'))"
```

## Installation and usage

```bash
# Install dependencies
pipenv install

# 1. Create the Pinecone index (once)
pipenv run python create_index.py

# 2. Ingest documents from ./fuentes/
pipenv run python add_sources.py

# 3. Start the search UI
pipenv run streamlit run search.py
```

The UI will be available at `http://localhost:8501`.

## Supported document formats

| Format | Processing |
|---|---|
| PPTX | Per-slide chunks preserving title and speaker notes |
| PDF | Text extraction with character-based chunking |
| DOCX | Paragraph extraction |
| XLSX | Row-by-row conversion as `column: value` |
| MP4 | Whisper transcription with time-aware overlapping chunks |

## Docker deployment

```bash
# Build image (includes only search.py)
docker build -t aesthetic-diagnosis-search .

# Run passing environment variables
docker run -p 8080:8080 \
  -e PINECONE_API_KEY=... \
  -e PINECONE_INDEX_HOST=... \
  -e OPENAI_API_KEY=... \
  aesthetic-diagnosis-search
```
