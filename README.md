# AskDocs

Try [**AskDocs**](https://johnaronb-askdocs.streamlit.app)
A RAG app that lets you ask questions about Anthropic research papers. It retrieves relevant
sections from the papers and uses Claude to answer based on what it finds.

---

## Requirements

- Python 3.10+ (The streamlit app specifically uses Python 3.12)
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

Download research papers as PDFs and place them in the `docs/` folder using the naming
convention `yyyy-Mmm-dd_paper-name.pdf`. Then run:

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

The app builds the index automatically on first startup if one does not exist yet.

---

## Deploying to Streamlit Community Cloud

1. Push the repo to a public GitHub repository
2. Go to share.streamlit.io and connect the repo
3. Fill out details and on the Advanced Settings' secret settings add `'ANTHROPIC_API_KEY = "your-key-here"`.
4. On first boot, the app will build the index from the PDFs in the `docs/` folder automatically

---

## Notes

- Questions are capped at 250 characters.
- Set a monthly spend limit on your Anthropic account as a cost safety net.
- The research papers in `docs/` are the intellectual property of Anthropic, PBC and are
  included here for educational purposes only. All rights to those documents remain with
  their respective authors and Anthropic.

---

## License

The source code is licensed under the MIT License. See LICENSE for details.
The research papers in `docs/` are not covered by this license.
