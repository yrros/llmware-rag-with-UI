# RAG Workspace Manager

A local semantic RAG (Retrieval Augmented Generation) search workspace for notes and documents, built with [LLMWare](https://github.com/llmware-ai/llmware), [LanceDB](https://lancedb.com/), [Streamlit](https://streamlit.io/), and optional [Ollama](https://ollama.com/) summarisation.

The app indexes markdown (and other supported) files from folder on your **local machine** (all local processing), embeds them with HuggingFace models, stores vectors in LanceDB, and lets you search by meaning—not just keywords, As well as reducing the search with a keyword search (like the year). You can optionally send retrieved chunks to a local LLM for a natural-language answer.

Each subfolder under your corpus root becomes its own **library** (e.g. `Personal Projects`, `Book Notes`, `Development`). You can select the embedding model for each of these folders to get the right level of precision.
This approach means you can search those top-level folders quickly.

This solution was built after trying other RAG solutions including Obsidain with copilot (slow to index and no GPU use), LLM Anyware was also tried but found it could not harness the GPUS and multi core CPUs of a modern Mac.
The system used for this was a M3 Pro with 36GB of RAM.

## What it does

1. **Ingest** — Scans a corpus folder recursively (including subfolders) and loads files into an LLMWare library.
2. **Embed** — Builds a semantic vector index using a configurable local embedding model.
3. **Search** — Runs hybrid retrieval: semantic vector search with optional keyword pre-filtering and text-search fallback.
4. **Answer (optional)** — Sends top search results to Ollama to synthesise an answer grounded in your notes.


---

## Architecture

```
Apple Notes export (markdown folders)
        │
        ▼
  LLMWare Library  ──►  SQLite blocks (text chunks)
        │
        ▼
  Embedding model  ──►  LanceDB vectors  (./llmware_data/lancedb/)
        │
        ▼
  Streamlit UI  ──►  Semantic search  ──►  Ollama (optional answer)
```

| Component | Role |
|-----------|------|
| `UI-semantic-search.py` | Streamlit app — indexing, search, settings |
| LLMWare | Library management, parsing, embedding, query API |
| LanceDB | Persistent vector store for semantic search |
| Ollama | Local generative LLM for “Answer with LLM” |
| `./llmware_data/` | All indexes, settings, and model cache (local to project) |

---

## Prerequisites

- **Python 3.10+** (3.13 tested; 3.14 works with app patches)
- **Apple Silicon Mac** (M-series) — configured for `mac_metal` / MPS; CUDA used automatically on NVIDIA machines
- **Virtual environment** with dependencies installed — see [Setup](#setup) (`pip install -r requirements.txt`)
- **Ollama** (optional) — for generative answers; on Mac, [install via Homebrew](#2-install-ollama-on-mac-optional) (recommended)
- **Exported notes** — markdown files organised in folders (see [Corpus layout](#corpus-layout)). Export from Apple Notes using the [Falcon Notes Exporter](https://falcon.star-lord.me/exporter).

---

## Setup

### 1. Clone and create a virtual environment

```bash
cd /path/to/rag_llmware
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install --requirement requirements.txt --upgrade-strategy eager
```

`pip install` resolves and installs **all transitive dependencies** automatically (for example `numpy`, `pillow`, `safetensors`, and `tokenizers` alongside `torch` / `torchvision` / `transformers`). Do **not** use `--no-deps`.

Or install the top-level packages manually (pip still pulls their dependencies):

```bash
pip install llmware streamlit lancedb torch torchvision transformers pyarrow pydub
```

On **Python 3.13+**, also install:

```bash
pip install audioop-lts
```

Use the same `venv` for every run. The app expects `lancedb` to be available in the active environment.

### 2. Install Ollama on Mac (optional)

For **Answer with LLM**, install [Ollama](https://ollama.com/). On a Mac, the easiest route is [Homebrew](https://brew.sh) — the standard package manager for macOS (installs Ollama and its dependencies in one step).

If you do not have Homebrew yet, install it from [https://brew.sh](https://brew.sh), then run:

```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
```

`start_app.sh` will also try to start Ollama if it is not already running.

### 3. Point the app at your note export

**In the app:** use **Notes location** at the top of the left sidebar. Enter the path to your export folder or click **Browse…** to pick it. The path is saved to `./llmware_data/app_settings.json`.

**Export from Apple Notes:** use the [Falcon Notes Exporter](https://falcon.star-lord.me/exporter) to produce a folder of markdown files (typically with an `iCloud` subfolder containing one directory per note account or category).

Each **immediate subfolder** of the corpus root becomes a selectable library in the UI—for example:

```
/your/export/iCloud/
├── Work Notes/
├── Book Notes/
└── Development/
```

Set the corpus root to `/your/export/iCloud` (or whichever folder contains your library subfolders). There is no default — the path must be set in the UI or saved in `app_settings.json`.

**Corpus mode** (sidebar radio, saved in `app_settings.json`):

| Mode | Use when |
|------|----------|
| **Multi-Corpus** | Organised export with one subfolder per collection (default) |
| **Single-Corpus** | Smaller or flat exports — one searchable index for every file under the root |

### 4. (Optional) Desktop launcher

```bash
./Installer.sh
```

Creates `~/Desktop/LLM-RAG.app`, which opens Terminal, activates `venv`, and starts Streamlit from the project directory.

---

## Running the app

Always start from the **project root** so data lands in `./llmware_data/`:

```bash
cd /path/to/rag_llmware
./start_app.sh
```

Or manually:

```bash
cd /path/to/rag_llmware
source venv/bin/activate
python3 ensure_ollama_ready.py   # optional; start_app.sh runs this automatically
python3 -m streamlit run UI-semantic-search.py
```

`start_app.sh` (and the Desktop launcher from `Installer.sh`) starts **Ollama if it is not already running**, then warms the **Answer with LLM** model configured in your corpus settings (most common `ollama_model` across libraries, usually `llama3.2:3b`). Embedding models are **not** pre-loaded at startup.

Re-run `./Installer.sh` once if you already have an older Desktop launcher.

Streamlit opens in your browser (typically `http://localhost:8501`).

---

## Corpus layout

Export your notes with the [Falcon Notes Exporter](https://falcon.star-lord.me/exporter), then point the app at the folder that contains your library subfolders.

Your export directory should look like:

```
/path/to/notes/iCloud/
├── Work Notes/
│   ├── meeting-notes.md
│   └── subfolder/
│       └── more-notes.md
├── Book Notes/
│   └── ...
└── Development/
    └── ...
```

Supported ingest extensions include `.md`, `.txt`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, and `.json`.

---

## Operation guide

### First-time indexing

1. Set **Notes location** to your Apple Notes export folder (or use **Browse…**).
2. Select a **library** (corpus) in the left sidebar.
2. Open **Advanced — Indexing & Embeddings** if you want to change the embedding model or relevance cutoff.
3. Click **Re-index (full rebuild)** for a new corpus, or **Rescan (incremental)** to add new/changed files.

Progress appears in the sidebar: a live progress bar during the scan, then a persistent summary (docs, blocks, vectors, model used).

**First index** does not require the “I understand rebuild” checkbox. That acknowledgement only appears when changing the embedding engine on an existing index.

### Search

1. Choose your library.
2. Optionally set a **Pre-filter keyword** to narrow to notes containing specific text.
3. Enter your question in **What would you like to ask?**
4. Click **Search**.

Results appear as expandable **Sources** with links to the original markdown files when paths can be resolved.

### Answer with LLM

- **Answer with LLM** is enabled by default.
- Configure **Ollama model**, **Max tokens** (slider), and **Temperature** below the search bar.
- Search runs first; Ollama summarises the retrieved chunks—it does not replace retrieval.

Requires Ollama running locally (`localhost:11434`).

### Advanced settings

| Setting | Scope | Notes |
|---------|-------|-------|
| Also use GPU for embeddings | Global | Uses CUDA when available. On Apple Silicon, HF embeddings stay on CPU due to a PyTorch MPS bug—leave off for best speed. |
| Embedding engine | Per corpus | Changing model requires **Re-index** after acknowledging the warning. |
| Relevance cutoff (distance) | Per corpus | Lower = stricter matches. Saved per corpus; no re-embed needed. |
| Debug details | Per session | Paths, file counts, index health |

### Rescan vs Re-index

| Action | When to use |
|--------|-------------|
| **Rescan (incremental)** | New or updated files; same embedding model |
| **Re-index (full rebuild)** | First index, corrupted index, or embedding model change |

---

## Data storage

All application data lives under **`./llmware_data/`** in the directory you run Streamlit from:

```
llmware_data/
├── lancedb/              # Vector indexes (LanceDB)
├── accounts/             # LLMWare libraries (SQLite + parsed blocks)
├── model_repo/           # Downloaded embedding models
├── corpus_settings.json  # Per-corpus embedding model, cutoff, Ollama prefs
├── job_status.json       # Last scan summary per corpus
└── app_settings.json     # Global settings (GPU toggle)
```

This folder is gitignored. To reset everything:

```bash
rm -rf llmware_data
```

Then re-index your corpora. Your source markdown files on disk are not deleted—only local indexes and settings.

On first run, the app may copy data from the legacy location `~/llmware_data/` if the project-local folder is empty.

---

## Configuration reference

### Per-corpus (`corpus_settings.json`)

- `embedding_model` — e.g. `mini-lm-sbert`, `gte-large`
- `distance_threshold` — semantic distance cutoff (default `0.55`)
- `ollama_model`, `llm_max_tokens`, `llm_temperature` — LLM answer preferences

### Global (`app_settings.json`)

- `corpus_root_path` — top-level folder containing library subfolders
- `corpus_mode` — `multi` (default) or `single`
- `use_gpu_for_embeddings` — default `false`; CUDA only on supported hardware

### In-app constants (`UI-semantic-search.py`)

- `DEFAULT_EMBEDDING_MODEL` — `mini-lm-sbert`
- `OLLAMA_HOST` / `OLLAMA_PORT` — default `localhost:11434`

---

## Recommended embedding models

| Model | Trade-off |
|-------|-----------|
| `mini-lm-sbert` | Fastest default |
| `bge-small-en-v1.5` | Better quality, still reasonably fast |
| `gte-large` | Strong retrieval, slower indexing |

Use **Show all local embedding models** in Advanced to see the full catalog.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| Wrong or missing corpus root | Path not set or folder moved | Set **Notes location** in the sidebar; use **Browse…** |
| Search returns nothing | Index empty or stale LanceDB path | Run **Re-index**; check Debug details for vector count |
| Only a few `.md` files indexed | Old LLMWare single-level scan | This app uses recursive ingest—run **Re-index** |
| Ollama answer fails | Ollama not running or model not pulled | `ollama serve` and `ollama pull llama3.2:3b` |
| MPS / GPU embedding error | PyTorch MPS bug with HF embeddings | Leave **Also use GPU** off on Mac (default) |
| Empty index after rebuild | Run from wrong directory | Always `cd` to project root before starting Streamlit |
| Slow indexing | Large corpus + heavy embedding model | Use `mini-lm-sbert` or `bge-small-en-v1.5` |

---

## Project files

| File | Purpose |
|------|---------|
| `UI-semantic-search.py` | Main Streamlit application |
| `start_app.sh` | Start/warm Ollama, then launch Streamlit |
| `ensure_ollama_ready.py` | Starts Ollama if needed and warms the configured LLM |
| `Installer.sh` | Creates the Desktop `LLM-RAG.app` launcher with the same startup steps |
| `.gitignore` | Excludes `llmware_data/`, venvs, `__pycache__` |

---

## License

Depends on upstream packages (LLMWare, Streamlit, LanceDB, HuggingFace models, Ollama). Check their respective licenses for redistribution and commercial use.
