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

   **Quick dates** sit on the heading line of **Must include**, **Any of** and **Exclude**, and put date terms into that field. Clicks add up until you search, so **Today** then **Yesterday** in **Any of** finds notes from either day. After a search, the next click first takes out the dates that were in that field for the search (and dates restored when the app starts), whether a button added them or you typed them (`2026-10-25`, `2026-10`, `2026`, `2026/10/25`, `25/10/2026`, `Week#43`…). Other terms, and the other two fields, stay as they are:

   | Button | Adds to Must include (on Sunday 25 October 2026) | Adds to Any of / Exclude |
   |---|---|---|
   | **Today** | `2026-10-25` | the same |
   | **Yesterday** | `2026-10-24` | the same |
   | **This week** | `2026` and `Week#43` | the 7 dates `2026-10-19` … `2026-10-25` |
   | **Last week** | `2026` and `Week#42` | the 7 dates `2026-10-12` … `2026-10-18` |
   | **This month** | `2026-10` | the same |
   | **Last month** | `2026-09` | the same |

   Days match notes that contain the date as `YYYY-MM-DD`, and months every date in that month. In **Must include**, weeks match notes headed `Week#NN` in that year, using ISO weeks (Monday to Sunday; 29 Dec 2025 is 2026 `Week#1`). The year is its own term rather than a month, so a week that spans two months still matches. Because terms match anywhere, `Week#4` also matches `Week#40`–`Week#49`, which only matters in weeks 1–9. In **Any of** and **Exclude** a bare year would match, or exclude, the whole year, so weeks are the seven days instead. A bare four-digit number counts as a year, so a button would take a leftover one out too. Hover a button to see exactly what it puts in.
3. Enter your question in **What would you like to ask?**
4. Click **Search**.

The search area is split into a **Filter** box (left) and an **Ask** box (right).

**Search route.** The switch at the top of the Filter box picks one of two routes into your notes; the Filter box shows only the settings for that route, and the Ask box's button follows it:

| Route | What **Search** does | Settings shown |
|-------|----------------------|----------------|
| **Semantic (meaning)** | Ranks passages by meaning using the embeddings. Filter terms narrow which notes it looks in. | **Distance cutoff** (how close a match must be) and, after a search, the match bar. |
| **Literal (exact text)** — button reads **Search exact text** | Returns the notes that match the filter terms, in full. Much faster; the question is only used for the LLM answer. | **Search all libraries**, **🔎 Filter search only**, **🔄 Rescan → Filter search**. |

The route is saved between sessions. **🔎 Filter search only** (Literal route) lists the matching notes under the Filter box without an LLM answer, keeping your question for later. Tick **🚫** beside a listed note to exclude it from the next **Search** (semantic ranking and LLM answer); excluded notes stay in the filter list, crossed out, and **Clear exclusions** in the Filter box resets them for the session. A full **Search** shows the answer, sources and discussion full-width below both boxes.

On the Literal route, a note shows when the text index was last rescanned (for this library, or every visible library when **Search all libraries** is ticked) and whether auto-rescan is on. **🔄 Rescan → Filter search** picks up new, edited and deleted notes first, then runs **Filter search only**. It is text only, so the filter sees the changes straight away; semantic search sees them after **⚡ Rescan**.

The **Literal** route skips vector search and returns the matching notes in full (the question is optional and only used for the LLM answer). Add **Search all libraries** to run that literal search across every visible library: a *Matches by library* panel shows note counts per library, with a **Use library** button to switch to one for a semantic search. No embeddings are used, so nothing needs re-indexing.

Results appear under **Sources** as one dropdown per document (documents with several matches are grouped). Each dropdown has two tabs:

- **Matched passages** — the retrieved chunks with their rank and distance.
- **Full document** — the whole file embedded in the page (markdown is rendered; PDF/DOCX/PPTX text is rebuilt from the indexed blocks), plus a download button for the original.

### Answer with LLM

The Ask box reads top to bottom: your question → the injected prompt's **links** (if it has any) → **Answer with LLM** → **Prompt library** → **Ollama model details** → **Search** → progress. Search settings are in the Filter box: **Max results** under **Exclude (NOT)**, and **Distance cutoff** in **Advanced search settings**.

- **Answer with LLM** is enabled by default.
- **Prompt library** — see [Prompt library](#prompt-library) below.
- **Ollama model details** is folded by default; a line under it shows the model, max tokens (with how long that would take at the model's last measured speed) and temperature in use. Inside: **Ollama model**, **Max tokens** and **Temperature** (saved per corpus as soon as you move them).
- **Max tokens** goes up to 50,000 (≈37,500 words). Its stops include book lengths for scale — *The Very Hungry Caterpillar*, *Where the Wild Things Are*, *The Tale of Peter Rabbit*, Dr. Seuss's *The Cat in the Hat*, *The Little Prince*, *Winnie-the-Pooh*, *The Old Man and the Sea*, *Animal Farm* and *The Lion, the Witch and the Wardrobe* — and the label and the line under the slider say how long the current setting is (word counts are approximate; English runs at about 0.75 words per token). *The Hobbit* (≈127,000 tokens) and *War and Peace* (≈783,000) are beyond the limit and shown for scale. The limit includes a reasoning model's thinking. Long answers take time (at 20 tokens/s, 50,000 tokens is about 40 minutes) and use more memory: Ollama's context window grows to fit the prompt plus the answer (8k, 16k, 32k or 64k tokens).
- **Max results** (in the Filter box, under **Exclude (NOT)**) caps how many passages (Semantic) or notes (Literal) a search returns.
- **Distance cutoff** (in the Filter box's **Advanced search settings**, Semantic route only; same setting as *Relevance cutoff* in the left panel, saved per corpus) decides which passages count as semantic matches, so it applies with or without an LLM answer. After a semantic search, a bar under it shows where passages were found — green is within the cutoff, amber is where to raise it — with a **Set cutoff to …** button to include the closest match.
- **Progress** shows right under **Search**: a spinner with the elapsed time while searching and answering, and beside it the generation speed (tokens per second and tokens so far), refreshed every 3 seconds. Answers stream from Ollama, so there's no overall time limit. The answer caption ends with the final tokens per second.
- **Speed score.** After each answer the model's speed is saved (`llm_speed` in `app_settings.json`, per model) and shown under **Search** between runs, e.g. *Last run with `qwen3:30b` (25 Oct 10:42): 12.4 tokens/s · 10,000 tokens ≈ 13 min 26 s*, for whichever model is selected.
- Search runs first; Ollama summarises the retrieved chunks—it does not replace retrieval.
- **Folder paths.** Each result gets its note's folder path from where the file sits under the notes folder, e.g. *Clients > Prospects > Acme*. It's shown on the note in **Sources** and given to the LLM with every passage (`[Source 2: Kickoff-2026-10-20.md, in Clients > Prospects > Acme]`), so the model knows what each note is about even for passages that don't include its title. No rebuild is needed. When two notes in a library share a file name, the path is left out rather than guessed. Local Apple Notes Export can also put the path in each note's title line (its *Folder path in titles* option), which puts it in the indexed text for search, after a rescan and re-embed.

### Prompt library

Open it with the **Prompt library** button in the left panel (hover it for a summary of what is switched on), or **Manage** in the Ask panel's Prompt library box. It opens in a new tab (`?page=prompts`) and saves to `llmware_data/prompt_library.json` (the corpus prompt stays in `corpus_settings.json`).

| Section | What it does |
|---------|--------------|
| **Common links** | Folded, at the very top. First, the **⚙️ Task executor**: the service that processes the outbox (e.g. the Salesforce Connector app at `http://localhost:8503`). It's defined in its own row, can be cleared but not removed, is listed first wherever links are, and is shown at the top of the **Actions** page with its 🟢/🔴 status. Then **🔗 Other links**: short-titled links (e.g. *SF Tasks*, *Connector*) that you attach to prompts with each card's **Links** selector. While a prompt is injected, its links show as hyperlinks between the question and the prompt token, opening in a new tab. Links to this computer (localhost) show 🟢 when the service answers and 🔴 when it doesn't (checked at most every 15 seconds). **Best guess search:** put `{best_guess}` in an address (e.g. a Salesforce search URL) and it's replaced with the app's best guess at what you're looking for, shown in the link text too (*Find in SF: Acme*): the Agent's staged or sent *Related to* names, else the first **Must include** filter term, else capitalised names in your question. Stored in `prompt_library.json`. |
| **Outbox** | Collapsed near the top. The folder that sent drafts are written to, and how many packages are waiting, processed and failed. See [Agents and the outbox](#agents-and-the-outbox). |
| **Task title marker** | Text added **before** or **after** the title of everything an Agent stages (task subjects, note titles, email subjects), e.g. `RAG_TEST:` or `[AI]`, so those records are easy to find later. Applied when the Agent stages, so you see it before **That's right**; titles that already contain it are left alone. Empty means off. |
| **About me** | One free-text box sent with every LLM request in every corpus — your name, where you're based, accessibility needs, how you like answers. Untick **Include in LLM requests** to switch it off. Today's date is always added. **Short name** (e.g. `JDOE`) signs the updates Agents make, such as a Next Step (*2026-10-25-JDOE: …*); it isn't sent to the LLM. |
| **Corpus prompt** | Pick a corpus and describe the role you wrote its notes in (e.g. business owner, salesperson). Tick **Apply corpus prompt** to send it for that corpus only. |
| **Task prompts** | Reusable instructions such as "Write a meeting summary and next steps as an email". Each has a name, text, **Global**, **Starred** and **Agent** checkboxes, a **Model** box, tags (for finding prompts) and a delete button. Edits save as you go, with a *Saved* time under the text. Tick **Global** to send a prompt with every request in every corpus, like About me — for standing instructions. Tick **Agent** to make it a workflow that ends by sending to the outbox (see below); Agent prompts have a blue border. **Model** picks the Ollama model used when you choose the prompt (see below). Above the list: full-text search, **Clear search**, an **Agents only** switch, and **Common keywords** (tags from your most recently used prompts) to filter with one click. |
| **Add a prompt** | Name, text, **Global**, **Starred**, **Agent**, the Agent's output, **Model** and tags, then **Add prompt**. Add an Agent's steps and Action Command on its card afterwards. |

**Model per prompt.** Leave it on *Model: as selected* to use whatever is chosen in the Ask panel. Pick a model (any you have pulled into Ollama, e.g. a Qwen model) and choosing the prompt selects it as the **Ollama model** for the answer and as the **Follow-up model** for the discussion. You can still change either one in the Ask panel; that overrides it for the run. A model picked by a prompt doesn't become the corpus's saved default.

**Agent prompts** are the only prompts that send anything to the outbox. Each is a workflow you refine as you use it:

| Part | What it is |
|------|------------|
| **Step 1 · Collection** | The prompt text. It runs with the search: what to find in the notes, list (numbered, so you can pick by number) and ask about. |
| **Steps 2, 3, …** | Optional. Each has a short name and a prompt, e.g. *Select tasks* ("List the tasks again and ask which to send") or *Set due dates*. They become one-click buttons in the Discussion panel. **Add step**, reorder with the arrows, remove with ✕. |
| **Output** | What the Agent sends: **Note**, **Tasks** or **Email** (the box next to **Agent**). |
| **Action Command** | The last step, e.g. *Send to tasks in Salesforce*. It's the instruction used to stage the result. |

A prompt can be Global or Agent, not both. Prompt libraries saved before Agent workflows existed are upgraded on load: prompts already set to draft a note, tasks or an email become Agents with no middle steps and a default Action Command (*Create these as tasks*, *Save this as a note* or *Draft this as an email*).

Four example task prompts are added the first time; edit or delete them freely.

**In the Ask panel** (shown when **Answer with LLM** is on), the compact **Prompt library** box sits above the model settings and has:

- **Always** — About me, the corpus prompt and each global prompt, ticked. Untick one to leave it out of this run.
- **Task** — click a prompt to attach it (click it again, or **✕**, to remove it); your question still decides which notes are found, and the task tells the LLM what to write from them. The task also applies to follow-up questions. On the **Literal** search route, a task can produce an answer even without a question.
- **Recent** — titles of the five task prompts most recently used for an answer, newest first. **Starred** — up to five starred prompts not already in Recent, most recently used first, so favourites fill the gap. Hover a title to see the full prompt.
- **Find a prompt** — search every task prompt by name or tag.
- **Agent prompts** read *Agent:Name* (e.g. *Agent:SF_Tasks*) and have a blue border: on the Recent and Starred buttons, on the chosen **Task**, and on the prompt token under the question box. **Find a prompt** shows the same *Agent:* names.

Each request is sent in this order: today's date → About me → corpus prompt → global prompts → notes and question → task. The answer caption lists the prompts that were used.

Requires Ollama running locally (`localhost:11434`).

### Agents and the outbox

Choose an Agent as the task and **Search**. Its collection prompt (step 1) writes the first answer: a numbered list of what it found, and questions. Then, in the **Discussion** panel:

1. **Refine.** Reply as usual, e.g. *Yes:1,2,3 ignore 4,5,6*, or give due dates and owners. The Agent box under the discussion has a button for each of the Agent's steps (*2 · Select tasks*, *3 · Set due dates*…) that sends that step's prompt for you. **Edit prompt workflow** opens this Agent in the prompt library in a new tab, so you can improve its steps as you go; changes apply on your next click. Everything in the discussion can be copied: each message has **Copy**, a step or **Needs work** message also has **Copy prompt sent** (the full text the model received), and **Copy task prompt** under the discussion caption (and beside **Copy answer**) copies the task the model was given, ready to reuse in a flow or a test.
2. **Narrow the sources.** Untick a note under **Sources** to leave it out of the following replies and of what the Agent stages. The discussion caption says how many notes are in use.
3. **Stage.** Click **Agent:Name** (blue outline; hover it to see the Action Command). The LLM turns what you agreed into exact records and shows them in a staging box with a one-line summary. Nothing has been sent yet. Remove any record with ✕.
4. **That's right** writes the staged records to the outbox as one package (checked against `schemas/handoff.v1.schema.json`); each record in it becomes its own task, note or email in the other system. Under each sent package, **View in outbox** opens the Actions page in a new tab, filtered to that package. **Needs work** discards them and asks what's wrong; your explanation goes into the discussion (with the rejected list, so the LLM knows what to fix), and you carry on refining and stage again.

Unknown values stay empty; nothing is guessed. **Related to** is a name (e.g. *Acme pilot*), not a link; the receiving system finds the matching record. You can stage and send more than once in the same discussion; each **That's right** is a new package.

**Reference snapshot (real record names).** If the task executor provides one, `outbox/reference/opportunities.json` (format `llm-notes-reference/v1`, see `schemas/reference.v1.schema.json`) lists current records from the other system, e.g. the Salesforce connector's open opportunities with recent activity: name, account, stage, last updated, last activity and description. This app only reads it; it never contacts the other system. With a snapshot:

- the Agent is given the real names and told to use one exactly in **Related to** (or leave it empty rather than invent one);
- in the staging box each item shows **✓** with the record's account, stage and last update when it matches, or **⚠** when it doesn't (the connector would hold it), and a picker to choose the right record (notes must be linked; tasks can also be standalone);
- **best guess search** links prefer a record whose name or account appears in your question;
- the Agent box says how many records the snapshot holds and when it was refreshed. Refresh it from the task executor (the Salesforce Connector app refreshes it on every scheduled run and with **Refresh now**).

The snapshot holds customer data, stays in the gitignored outbox folder and is only sent to your local Ollama model. The starter Agent **SF_Opp_Notes** (in the prompt library) summarises what your notes say about each opportunity and stages one note per chosen opportunity, linked to it in Salesforce.

**Next step + chase task.** An Agent with this output updates an opportunity's **Next Step** and creates a task to chase it. For each opportunity it stages two items in one package:

- **Next Step**, signed with today's date and your **Short name** (About me), plus the task title marker, e.g. *2026-10-25-JDOE: waiting on client to respond, will chase on 29 October [AI]*. The staging box shows the new text and the current Next Step from the snapshot (*Now: …*), and warns above 255 characters (the most a Next Step holds) or when no short name is set.
- **A chase task** on the same opportunity, with the due date from your notes or discussion, and the marker.

The connector writes the Next Step only if the opportunity still has the one you saw; if someone changed it since, it holds the update as *changed since staged* (refresh the snapshot and stage again). These packages use format `llm-notes-handoff/v2` (`schemas/handoff.v2.schema.json`): v1 plus update items and mixed packages. On the Actions page a waiting Next Step can be edited like any item. The starter Agent **SF_Next_Step** asks which opportunities to update and when to chase.

The app never contacts another system. A separate connector (for example [rag-action-interface](https://github.com/yrros/rag-action-interface) for Salesforce) reads the outbox, moves each package to `outbox/processed/` or `outbox/failed/`, and leaves a `<package name>.receipt.json` beside it. The Agent box shows each sent package's result once its receipt appears: what was created, and why anything wasn't.

**Outbox status line.** The top of every page shows, e.g., *Outbox: 2 waiting (7 items) · 1 needs attention · Last processed 25 Oct 10:42 · Last error 24 Oct 18:03 · Open Actions*. "Waiting" counts packages not yet picked up and the records in them; "needs attention" counts packages in `failed/`, including items held for you to choose a record; the dates come from the connector's receipts.

**Actions page.** **Open Actions** at the end of the status line opens `?page=actions` in a new tab: everything in the outbox, newest first, in four tabs — **Needs attention** (packages in `failed/`, opened up), **Waiting**, **Processed** and **Withdrawn**. Each package shows the Agent that sent it, its library, question, model and source notes, the connector's overall result, and every record with its outcome: the record ID it created, or why it didn't (with possible matches when a name was ambiguous). Filter by any text, e.g. a subject, owner or client name; **Refresh** re-reads the folder. Under the title, **Task executor** links to the service that processes the outbox (set in the prompt library's **Common links**), with 🟢 running / 🔴 not running.

**Getting back.** On every screen other than the main one (the Prompt library, the Actions page and the Salesforce Connector app) the 🧠 before the title takes you back to the main screen in the same tab.

| Button | On | What it does |
|--------|----|--------------|
| **Withdraw** | Waiting, Needs attention | Asks to confirm, then moves the package (and its receipt) to `outbox/withdrawn/`, where connectors don't look. Nothing already created in the other system is undone. |
| **Retry** | Needs attention | Removes the receipt and moves the package back to the outbox for the connector's next run, e.g. after a temporary error. Records already created aren't created twice. Disabled when the file itself is invalid. |
| **Restore** | Withdrawn | Moves the package back to the outbox for a connector to process. |
| **Edit** | Waiting, Needs attention, Withdrawn | Opens the package's items for editing: subject or title, description or body, due date, priority, owner, recipients and **Related to**. **Save changes** checks the package against the schema and rewrites the file atomically, keeping its id (so nothing is ever created twice). Use it to add a marker such as `[TEST]` to find records easily, or to fix a name the connector couldn't match, then **Retry**. On a partly done package, records already created aren't shown, since editing them wouldn't change the other system. If a connector picks the package up while you edit, nothing is saved. |

When a record needs choosing (an ambiguous or missing name), either choose it in the connector, which retries the package by itself, or correct **Related to** with **Edit** and press **Retry**. A package you withdraw also shows as withdrawn in the Agent box of the discussion that sent it.

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

Drafts you send are written to **`./outbox/`** next to it by default (also gitignored; see [Agents and the outbox](#agents-and-the-outbox)).

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
- `llm_speed` — per Ollama model, the tokens per second of its last answer (with when, and how many tokens), shown under **Search** as the time for 10,000 tokens

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
| `schemas/handoff.v2.schema.json` | Package format v2 (`llm-notes-handoff/v2`): v1 plus update items (e.g. a Next Step) and mixed packages |
| `schemas/reference.v1.schema.json` | Format of the read-only reference snapshot a connector writes to `outbox/reference/` (`llm-notes-reference/v1`) |
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
