# AskDocs

A simple RAG app that lets you ask questions about Anthropic research papers. It pulls relevant
sections from the papers and uses Claude to answer based on what it finds.

---

## Requirements

- Python 3.10+
- An Anthropic API key (get one at platform.anthropic.com)

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the secrets file:

```bash
mkdir -p .streamlit
echo 'ANTHROPIC_API_KEY = "your-key-here"' > .streamlit/secrets.toml
```

Replace `your-key-here` with your actual key. This file is in `.gitignore` and will never
be committed.

---

## Adding papers

Download research papers as PDFs and place them in the `docs/` folder. Then run:

```bash
python ingest.py
```

This reads the PDFs, splits them into chunks, and saves a local vector index under `chroma_db/`.
Run this again whenever you add or remove papers.

---

## Running the app

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

The app builds the index automatically on first startup if one doesn't exist yet.

---

## Deploying to Streamlit Community Cloud

1. Push the repo to a public GitHub repository
2. Go to share.streamlit.io and connect the repo
3. Add your `ANTHROPIC_API_KEY` under the app's Secrets settings
4. On first boot, the app will build the index from the PDFs in the `docs/` folder automatically

---

## Notes

- Questions are capped at 250 characters
- Set a monthly spend limit on your Anthropic account as a cost safety net
