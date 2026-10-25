# RAG Workspace Manager

A local semantic RAG (Retrieval Augmented Generation) search workspace for notes and documents, built with [LLMWare](https://github.com/llmware-ai/llmware), [LanceDB](https://lancedb.com/), [Streamlit](https://streamlit.io/), and optional [Ollama](https://ollama.com/) summarisation.

The app indexes markdown (and other supported) files from folder on your **local machine** (all local processing), embeds them with HuggingFace models, stores vectors in LanceDB, and lets you search by meaning—not just keywords, As well as reducing the search with a keyword search (like the year). You can optionally send retrieved chunks to a local LLM for a natural-language answer.

Each subfolder under your corpus root becomes its own **library** (e.g. `Personal Projects`, `Book Notes`, `Development`). You can select the embedding model for each of these folders to get the right level of precision.
This approach means you can search those top-level folders quickly.

This solution was built after trying other RAG solutions including Obsidain with copilot (slow to index and no GPU use), LLM Anyware was also tried but found it could not harness the GPUS and multi core CPUs of a modern Mac.
The system used for this was a M3 Pro with 36GB of RAM.

## What it does

1. **Ingest** — Scans a corpus folder recursively (including subfolders) and loads files into an LLMWare library. Rescans pick up new, edited and deleted notes, and can run automatically in the background.
2. **Embed** — Builds a semantic vector index using a configurable local embedding model.
3. **Search** — Runs hybrid retrieval: semantic vector search with an optional boolean note filter (AND / OR / NOT) and text-search fallback, or a literal filter-only search across one or all libraries.
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

### Quick install (zip download)

1. Unzip the download (e.g. to `~/Downloads/llmware-rag-with-UI-main`).
2. Open Terminal, `cd` into that folder, and run:

```bash
cd ~/Downloads/llmware-rag-with-UI-main   # adjust path if different
chmod +x Installer.sh
./Installer.sh
```

`Installer.sh` creates a Python `venv`, installs dependencies from `requirements.txt`, and places **`LLM-Notes.app`** on your Desktop. Double-click that icon to start the app.

Keep the unzipped project folder where it is — the Desktop app remembers that path. If you move the folder later, run `./Installer.sh` again from the new location.

### 1. Clone and create a virtual environment

For a git clone (or if you prefer manual setup instead of `Installer.sh`):

```bash
cd /path/to/llmware-rag-with-UI
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

**First run:** the app opens a **Getting started** guide that walks you through choosing your notes folder, how to index it, and the embedding engine — see [Getting started guide](#getting-started-guide).

**In the app:** use **Notes location** at the top of the left sidebar. Enter the path to your export folder or click **Browse…** to pick it. The path is saved to `./llmware_data/app_settings.json`.

**Export from Apple Notes:** use our companion app [Local Apple Notes Export](https://github.com/yrros/Local-Apple-Notes-Export), or the [Falcon Notes Exporter](https://falcon.star-lord.me/exporter), to produce a folder of markdown files (typically with an `iCloud` subfolder containing one directory per note account or category).

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

If you did not use the [quick install](#quick-install-zip-download) above:

```bash
./Installer.sh
```

Creates `~/Desktop/LLM-Notes.app`, which opens Terminal, activates `venv`, and starts Streamlit from the project directory.

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
python3 -m streamlit run UI-semantic-search.py --server.address localhost
```

`start_app.sh` (and the Desktop launcher from `Installer.sh`) starts **Ollama if it is not already running**, then warms the **Answer with LLM** model configured in your corpus settings (most common `ollama_model` across libraries, usually `llama3.2:3b`). Embedding models are **not** pre-loaded at startup.

Re-run `./Installer.sh` if you move the project folder or need to refresh the Desktop launcher.

Streamlit opens in your browser (typically `http://localhost:8501`).

> **Security:** the app has no authentication and can display and download the full contents of your indexed documents. `start_app.sh` binds Streamlit to `localhost` so it is only reachable from your own machine. Do not expose it on a shared network or the internet.

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

### Getting started guide

On first run — and after `./reset_rag.sh` — the app shows a three-step guide instead of the normal screen. It appears whenever no corpus has been indexed yet.

1. **Choose your notes folder.** Use [Local Apple Notes Export](https://github.com/yrros/Local-Apple-Notes-Export) (our companion app), the [Falcon Notes Exporter](https://falcon.star-lord.me/exporter), or any other folder of Markdown (an Obsidian vault, a docs folder…). The guide counts the files it can index.
2. **Choose how to index:**
   - **Try a small corpus first** — fastest. Indexes one top-level folder (the smallest is preselected) so you can try search in minutes.
   - **Index everything as one corpus** — one search across all notes; slowest to build, and changing engine re-indexes everything. Sets **Single-Corpus** mode.
   - **Index each top-level folder as its own corpus** — most flexible; each folder gets its own database and can be re-indexed, hidden or switched to another engine on its own. Empty folders are skipped.
3. **Choose an embedding engine** (default `mini-lm-sbert`) and whether to use the GPU. Models marked **(g)** can use this machine's GPU.

**Start indexing** saves the choices and indexes straight away with progress on the page, then opens the app on the new library. **Skip for now** goes to the normal app for this session; the guide returns next time if nothing has been indexed.

### First-time indexing (manual)

1. Set **Notes location** to your Apple Notes export folder (or use **Browse…**).
2. Select a **library** (corpus) in the left sidebar.
2. Open **Advanced — Indexing & Embeddings** if you want to change the embedding model or relevance cutoff.
3. Click **Re-index (full rebuild)** for a new corpus, or **Rescan (incremental)** to pick up new, edited and deleted notes.

The sidebar groups these under **This corpus**. Below them, **All corpora** has **⚡ Rescan all corpora**, which rescans every visible corpus in turn (see [Rescan vs Re-index](#rescan-vs-re-index)).

Progress appears in the sidebar: a live progress bar during the scan, then a persistent summary (docs, blocks, vectors, model used).

**First index** does not require the “I understand rebuild” checkbox. That acknowledgement only appears when changing the embedding engine on an existing index.

### Search

1. Choose your library.
2. Optionally add filter terms as capsules (type a term, press Enter):
   - **Must include (AND)** — every term must appear in the note.
   - **Any of (OR)** — at least one term must appear.
   - **Exclude (NOT)** — notes with any of these terms are dropped.

   Terms match literally and case-insensitively anywhere in the note or its file name (`2026-10` matches `2026-10-19`). With a question, semantic search ranks passages only within the matching notes. Filters are saved between sessions.
3. Enter your question in **What would you like to ask?**
4. Click **Search**.

The search area is split into a **Filter** box (left) and an **Ask** box (right). **🔎 Filter search only** (in the Filter box's *Advanced search settings*) lists the matching notes under the Filter box without vector search or an LLM answer, keeping your question for later. Tick **🚫** beside a listed note to exclude it from the next **Search** (semantic ranking and LLM answer); excluded notes stay in the filter list, crossed out, and **Clear exclusions** in the Filter box resets them for the session. A full **Search** shows the answer, sources and discussion full-width below both boxes.

Below those, a note shows when the text index was last rescanned (for this library, or every visible library when **Search all libraries** is ticked) and whether auto-rescan is on. **🔄 Rescan → Filter search** picks up new, edited and deleted notes first, then runs **Filter search only**. It is text only, so the filter sees the changes straight away; semantic search sees them after **⚡ Rescan**.

**Exact text only** skips vector search and returns the matching notes in full (the question is optional and only used for the LLM answer). Add **Search all libraries** to run that literal search across every visible library: a *Matches by library* panel shows note counts per library, with a **Use library** button to switch to one for a semantic search. No embeddings are used, so nothing needs re-indexing.

Results appear under **Sources** as one dropdown per document (documents with several matches are grouped). Each dropdown has two tabs:

- **Matched passages** — the retrieved chunks with their rank and distance.
- **Full document** — the whole file embedded in the page (markdown is rendered; PDF/DOCX/PPTX text is rebuilt from the indexed blocks), plus a download button for the original.

### Answer with LLM

- **Answer with LLM** is enabled by default.
- Configure **Ollama model**, **Max tokens** (slider), and **Temperature** below the search bar. Max tokens and Temperature are saved per corpus as soon as you move them.
- **Distance cutoff** sits beside them (same setting as *Relevance cutoff* in the left panel, saved per corpus). After a semantic search, a bar under it shows where passages were found — green is within the cutoff, amber is where to raise it — with a **Set cutoff to …** button to include the closest match.
- Search runs first; Ollama summarises the retrieved chunks—it does not replace retrieval.
- **Prompt library** (above the model settings) — see [Prompt library](#prompt-library) below.

### Prompt library

Open it with the **Prompt library** button in the left panel (hover it for a summary of what is switched on), or **Manage** in the Ask panel's Prompt library box. It opens in a new tab (`?page=prompts`) and saves to `llmware_data/prompt_library.json` (the corpus prompt stays in `corpus_settings.json`).

| Section | What it does |
|---------|--------------|
| **Outbox** | Collapsed at the top. The folder that sent drafts are written to, and how many packages are waiting, processed and failed. See [Drafts and the outbox](#drafts-and-the-outbox). |
| **About me** | One free-text box sent with every LLM request in every corpus — your name, where you're based, accessibility needs, how you like answers. Untick **Include in LLM requests** to switch it off. Today's date is always added. |
| **Corpus prompt** | Pick a corpus and describe the role you wrote its notes in (e.g. business owner, salesperson). Tick **Apply corpus prompt** to send it for that corpus only. |
| **Task prompts** | Reusable instructions such as "Write a meeting summary and next steps as an email". Each has a name, text, **Global**, **Starred** and **Agent** checkboxes, tags (for finding prompts) and a delete button. Edits save as you go, with a *Saved* time under the text. Tick **Global** to send a prompt with every request in every corpus, like About me — for standing instructions. Tick **Agent** to make it draft output for the outbox (see below); Agent prompts have a blue border. Above the list: full-text search, **Clear search**, an **Agents only** switch, and **Common keywords** (tags from your most recently used prompts) to filter with one click. |
| **Add a prompt** | Name, text, **Global**, **Starred**, **Agent**, the Agent's output and tags, then **Add prompt**. |

**Agent prompts** are the only prompts that send anything to the outbox. Instead of a text answer, an Agent drafts a **Note**, **Tasks** or an **Email** (pick which in the box next to **Agent**; it defaults to Tasks) for you to review and send. A prompt can be Global or Agent, not both. Prompt libraries saved before Agent existed are upgraded on load: prompts that were already set to draft a note, tasks or an email become Agents.

Four example task prompts are added the first time; edit or delete them freely.

**In the Ask panel** (shown when **Answer with LLM** is on), the compact **Prompt library** box sits above the model settings and has:

- **Always** — About me, the corpus prompt and each global prompt, ticked. Untick one to leave it out of this run.
- **Task** — click a prompt to attach it (click it again, or **✕**, to remove it); your question still decides which notes are found, and the task tells the LLM what to write from them. The task also applies to follow-up questions. With **Exact text only**, a task can produce an answer even without a question.
- **Recent** — titles of the five task prompts most recently used for an answer, newest first. **Starred** — up to five starred prompts not already in Recent, most recently used first, so favourites fill the gap. Hover a title to see the full prompt.
- **Find a prompt** — search every task prompt by name or tag.

Each request is sent in this order: today's date → About me → corpus prompt → global prompts → notes and question → task. The answer caption lists the prompts that were used.

Requires Ollama running locally (`localhost:11434`).

### Drafts and the outbox

When the task prompt is an **Agent**, the LLM writes drafts instead of prose (the built-in **Action items** Agent drafts tasks). They appear under the answer as editable cards:

1. Review and edit each draft. Untick a draft to leave it out. Leave a field empty when the notes don't say; nothing is guessed.
2. **Related to** and **Kind** name the client, deal or person a draft belongs to (e.g. *Acme pilot*, *Opportunity*). They are a name, not a link; the receiving system finds the matching record.
3. Click **Send to outbox**. The drafts are checked against `schemas/handoff.v1.schema.json` and written as one JSON file, then locked so they can't be sent twice.

The app never contacts another system. A separate connector (for example [rag-action-interface](https://github.com/yrros/rag-action-interface) for Salesforce) reads the outbox, moves each package to `outbox/processed/` or `outbox/failed/`, and leaves a `<package name>.receipt.json` beside it. The drafts panel shows that result once the receipt appears: what was created, and why anything wasn't.

**Default location:** `outbox/` in the project folder, i.e. the folder you run Streamlit from, which is the project folder when you start with `start_app.sh` or the Desktop launcher. Change it under **Outbox** on the Prompt library page (saved as `outbox_path` in `app_settings.json`); the connector must read the same folder. The outbox is gitignored because sent drafts contain your notes, and `reset_rag.sh` leaves it alone.

Each file is named `YYYYMMDD-HHMMSS-<note|task|email>-<first 8 characters of the package id>.json` and written atomically (a hidden `.<name>.tmp` file is renamed into place).

### Advanced settings

| Setting | Scope | Notes |
|---------|-------|-------|
| Also use GPU for embeddings | Global | Uses CUDA when available, or the Apple Silicon GPU (MPS) on Mac. Falls back to CPU if a GPU op fails. |
| Auto-rescan every 3 min (text only) | Global | While the Streamlit server runs (with or without a browser tab), rescans the selected corpus for new, edited and deleted notes and updates the text index, so filters and literal search find them. Embeddings are not built — semantic search sees the changes after **⚡ Rescan**. Starts on the first page load after the server starts. |
| …in every corpus | Global | Auto-rescan every visible corpus instead of just the selected one. Hidden corpora are skipped. |
| Manage corpus visibility | Global | Hides corpora from the library list **and** from every all-corpora scan and rebuild, including auto-rescan. A hidden corpus keeps its index as it was; select it to rescan it on its own. Only applies while **Enable corpus hiding** is ticked. |
| Embedding engine | Per corpus | Changing model requires **Re-index** after acknowledging the warning. |
| Relevance cutoff (distance) | Per corpus | Lower = stricter matches. Saved per corpus; no re-embed needed. |
| Debug details | Per session | Paths, file counts, index health |

### Rescan vs Re-index

| Action | Scope | When to use |
|--------|-------|-------------|
| **⚡ Rescan (incremental)** | This corpus | New, edited or deleted notes; same embedding model. Embeds only the new blocks. |
| **🔨 Re-index (full rebuild)** | This corpus | First index, corrupted index, or embedding model change |
| **⚡ Rescan all corpora** | All visible corpora | Same as Rescan, for every visible corpus in turn, each on its own embedding engine. Nothing is wiped. |
| **🔨 Rebuild all corpora** (Advanced) | All visible corpora | Deletes and rebuilds every visible corpus on one embedding engine. |
| **🔨 Rebuild all corpora — text only** (Advanced) | All visible corpora | As above, but text only: filters work straight away, and each corpus needs **⚡ Rescan** to rebuild its embeddings. Rarely needed. |
| **Auto-rescan** / **🔄 Rescan → Filter search** | This corpus or all visible corpora | Text only, for literal search and filters. |

Hidden corpora are skipped by every all-corpora action.

**How changes are found.** When LLMWare indexes a note it copies the file into the library's `uploads/` folder. A note whose file is newer than that copy has been edited since: its old blocks and vectors are deleted and only that file is read again. Notes no longer on disk are removed from the index. Unchanged notes are never read, so a rescan with nothing to do takes well under a second. If a corpus folder is missing or empty, nothing is removed.

**Notes with the same file name.** LLMWare identifies a library's files by file name alone. If two notes in one corpus share a name (same title and creation date in different subfolders), only one of them is indexed.

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
├── app_settings.json     # Global settings (GPU toggle, auto-rescan, corpus root…)
├── text_rescan_status.json  # When each corpus's text index was last rescanned
└── prompt_library.json   # About me, task and global prompts
```

This folder is gitignored. To reset it, use `reset_rag.sh` (see [Resetting the environment](#resetting-the-environment)).

Drafts you send are written to **`./outbox/`** next to it by default (also gitignored; see [Drafts and the outbox](#drafts-and-the-outbox)).

On first run, the app may copy data from the legacy location `~/llmware_data/` if the project-local folder is empty.

---

## Configuration reference

### Per-corpus (`corpus_settings.json`)

- `embedding_model` — e.g. `mini-lm-sbert`, `gte-large`
- `distance_threshold` — semantic distance cutoff (default `0.55`)
- `ollama_model`, `llm_max_tokens`, `llm_temperature` — LLM answer preferences
- `follow_up_model` — Ollama model used in the Discussion panel (defaults to the answer model)
- `corpus_prompt`, `corpus_prompt_enabled` — your role as author of this corpus, sent to the LLM ahead of the notes when enabled
- `hidden` — default `false`; while corpus hiding is enabled, leaves this corpus out of the library list, **Search all libraries** and every all-corpora scan and rebuild

### Global (`app_settings.json`)

- `corpus_root_path` — top-level folder containing library subfolders
- `corpus_mode` — `multi` (default) or `single`
- `use_gpu_for_embeddings` — default `false`; CUDA only on supported hardware
- `auto_rescan_enabled` — default `false`; text-only rescan every 3 minutes in the background
- `auto_rescan_all_corpora` — default `false`; auto-rescan every visible corpus instead of just the selected one
- `hide_corpora_enabled` — default `false`; hide corpora marked `hidden` in `corpus_settings.json` from the library list and all-corpora scans
- `outbox_path` — folder that sent drafts are written to; empty means `outbox/` in the project folder

### In-app constants (`UI-semantic-search.py`)

- `DEFAULT_EMBEDDING_MODEL` — `mini-lm-sbert`
- `OLLAMA_HOST` / `OLLAMA_PORT` — default `localhost:11434`
- `AUTO_RESCAN_SECONDS` — auto-rescan interval, default `180`

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
| A new or edited note isn't found by the filter | Not rescanned yet, or its corpus is hidden | Click **🔄 Rescan → Filter search**, or turn on auto-rescan; unhide the corpus or select it and **⚡ Rescan** |
| A new or edited note is found by the filter but not by semantic search | Auto-rescan is text only | Click **⚡ Rescan** (or **⚡ Rescan all corpora**) to embed it |
| Only a few `.md` files indexed | Old LLMWare single-level scan | This app uses recursive ingest—run **Re-index** |
| Ollama answer fails | Ollama not running or model not pulled | `ollama serve` and `ollama pull llama3.2:3b` |
| MPS / GPU embedding error | PyTorch MPS op unsupported for a model | Untick **Also use GPU** and re-run the scan on CPU |
| Database cannot be started / SQLite errors after an interrupted scan | Scan stopped mid-write (battery, crash, force quit) and left the index corrupt | Quit the app, run `./reset_rag.sh`, then restart and re-index |
| Empty index after rebuild | Run from wrong directory | Always `cd` to project root before starting Streamlit |
| Slow indexing | Large corpus + heavy embedding model | Use `mini-lm-sbert` or `bge-small-en-v1.5` |

---

## Project files

| File | Purpose |
|------|---------|
| `UI-semantic-search.py` | Main Streamlit application |
| `start_app.sh` | Start/warm Ollama, then launch Streamlit |
| `ensure_ollama_ready.py` | Starts Ollama if needed and warms the configured LLM |
| `Installer.sh` | Creates `venv`, installs dependencies, and builds Desktop `LLM-Notes.app` |
| `reset_rag.sh` | Deletes indexes, vectors, histories and temp files; keeps models and settings |
| `schemas/handoff.v1.schema.json` | Format of the packages written to the outbox (`llm-notes-handoff/v1`), shared with connectors |
| `.gitignore` | Excludes `llmware_data/`, `outbox/`, venvs, `__pycache__` |

---

## License

Depends on upstream packages (LLMWare, Streamlit, LanceDB, HuggingFace models, Ollama). Check their respective licenses for redistribution and commercial use.

---

## GPU marker on embedding models

In the **Advanced — Indexing & Embeddings** model lists, a **(g)** in front of a model name means that model can embed on this machine's GPU. It is shown whether or not **Also use GPU for embeddings** is ticked; the GPU is only actually used for scans when that option is on.

A model is marked when:

- a supported GPU is available on this machine (CUDA, or MPS on Apple Silicon), and
- the model is a Hugging Face embedding model (all of the bundled local models are).

---

## Resetting the environment

If a scan is interrupted (battery dies, crash, force quit) the index can be left corrupt and the app may report that the database cannot be started. `reset_rag.sh` clears the local index data so you can start clean:

```bash
./reset_rag.sh             # list what will be deleted, then ask to confirm
./reset_rag.sh --dry-run   # only list what would be deleted
./reset_rag.sh --yes       # no confirmation prompt
./reset_rag.sh --settings  # also delete app_settings.json, corpus_settings.json and prompt_library.json
```

**Deleted** from `llmware_data/`: `accounts/` (SQLite text index and library folders), `lancedb/` (vector tables), `tmp/`, `parser_history/`, `query_history/`, `prompt_history/` and `job_status.json`.

**Kept:** downloaded models (`llmware_data/model_repo/` and `~/.cache/huggingface`), the `venv`, your source notes, and — unless you pass `--settings` — your corpus root, GPU toggle, corpus prompts, prompt library and per-corpus embedding choices.

Notes:

- Quit the app first; the script refuses to run while `streamlit run UI-semantic-search.py` is active (`--dry-run` works either way).
- It leaves empty `accounts/` and `lancedb/` folders so the app does not re-copy old data from the legacy `~/llmware_data/` on next start.
- Afterwards start the app with `./start_app.sh`; the [Getting started guide](#getting-started-guide) opens so you can re-index.
