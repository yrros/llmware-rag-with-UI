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

**In the app:** open **Configure** in the left sidebar's **Library Controls** (below **Prompt library**); its **Notes location** holds the corpus root folder and corpus mode. Enter the path to your export folder or click **Browse…** to pick it. Configure is folded once the folder is set, and opens by itself if it isn't, or can't be found. The path is saved to `./llmware_data/app_settings.json`.

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

**Oneshot search** (the box under the search page's heading): type what you want in one sentence, e.g. *Find notes from December 2024 where pricing was discussed in my Acme notebook*, and press Enter (or **Search**). **Start fresh**, to the right of **Search**, clears the page and starts again: everything **Clear all** clears, plus a running Note-Workflow and a Chat message's notes run (settings stay; anything already sent stays in the outbox). Shift+Enter adds a line; the box starts as one line and grows as the text wraps or lines are added. The mic beside it dictates the search instead (sent after a pause). It runs the same way as a notes question from Chat (see *Notes questions go to search*): read into a library, exact words, dates (*December 2024* becomes `2024-12`) or a topic by meaning, then find, keep, and ask, in the **Searching your notes** pop-up, with **Search again** if nothing matches. It's always treated as a notes search (no yes/no check). A short topic is looked for by its exact words first, then by meaning if none match; say *semantic*, *by meaning*, *similar to* or *related to* to search by meaning straight away (e.g. *find notes about holidays (use a semantic search), where was it discussed?*). Searching by meaning needs the library's semantic index (**Status: Ready**). A word that can be written as one word, two words or hyphenated (*heat pump*, *heatpump*, *heat-pump*) is looked for in every spelling, as alternatives, and a plural is looked for as its singular (*side-cars* finds *side-car*). For a search by meaning, the topic is put in plain, common nouns (*home heating* rather than a product name), which a meaning search matches far better. **Read the answer to me** is off by default for a typed one, remembered separately from dictation's; a spoken one follows dictation's (on by default). If a library's saved answer model isn't installed any more, it answers with an installed one (the chat's model, or `llama3.2:3b`) and says so once.

**Note-Workflows** (the menu under the Oneshot box) run multi-stage tasks on your notes. The menu lists starred workflows first (★), then the ones you ran most recently (up to five, newest first), then the rest. Beside it, each starred workflow has its own button, as many as fit the window's width (the rest stay in the menu), and **Add a Note-Workflow** sits on the right; it opens the Prompt library in a new tab with a new workflow ready to set up. **Weekly Kick-off** comes with the app, starred:

1. The search is set up afresh: **Clear all**, the Literal route, and **Any of** set to the days of the last working week (Monday to Friday of last week; from Thursday on, this week up to today, so you can prep it early). **Filter search only** runs, and every note found is kept in scope (**Max results** goes to 200 for this, without being saved).
2. **Stage topics** (a Checklist stage) asks the LLM to list the problems, plans and progress in those notes, grouped by client, prospect or subject, with each item's owner and due date where the notes give them. They're shown as a checklist, all ticked: untick what you don't want to take forward (**All** / **None** per group), and add anything to steer with in the **Context** box beside it. Above the list, **Group by** shows it by client / subject, person or kind; **Leave out** drops everything for the people you click (click again to bring them back); and **Change this list** takes an instruction in words (*exclude all actions for Sam*, *arrange these by person*, *merge duplicates*): the model rewrites the kept items (not the notes again), and **Undo** puts the last version back.
3. **Write the report** (a Written stage) gets the items you kept and your context, and writes the Problems, Plans and Progress report. **Copy** it, or **Send to LLM Chat** to refine it in a new chat.

A panel above the search shows the dates, the notes in scope and the stages (**Back**, **Next**, **Run again**, **Start again**, which clears the run and finds its notes afresh, and **Close**). **Next** runs the next stage straight away, with its prompt folded; while a stage runs, **Cancel** (floating at the bottom of the window) stops it, to correct the list or edit the prompt and press **Run**. An Agent stage shows its progress (tokens/s) in the panel, **Next** waits while it works, and once it has answered the workflow moves on to the next stage and runs it by itself; its answer and **Execute** stay open under the panel. When the next stage is another Agent (e.g. tasks, then opportunity notes, then Next Steps), it waits for you instead: press **Execute** and the items go into the hold and the next Agent runs, since its answer replaces this one (anything sent stays under **Held for review** on the Actions page). The stage line marks each Agent stage **✓ sent** or **answered, not sent**. A stage with nothing routed to it is skipped. On the same line as each stage's prompt, **Stop step** stops it while it runs (the same as **Cancel**), and **Skip and approve step** runs it and approves what it gives without stopping: a checklist goes on as it's routed, written output is kept, and an Agent's answer is executed (its items are still held for review, to drop any or **Recall** them); then the next step runs. Ticking it there also saves it to the workflow in the Prompt library. **Notes considered** lists every note the search found, newest first, ticked when it takes part: untick one to leave it out, or tick another to bring it in (the same as 🚫 in the list under the search); change the filter and press **Filter search only** to look elsewhere. Once the notes are found, the Filter and Ask boxes fold into **Filter and Ask** under the panel, and the search results into **Search results**, to keep the page clean; open either to change the search. They fold again at each new stage, except that once an Agent stage has run its answer stays open. Each stage's prompt is filled in for the run and can be edited there first; pick its model beside **Run**. Each stage is told what the next one does with its output, so it shapes what it writes for it.

Set workflows up in the Prompt library under **Note-Workflows**: the name, **Favourite** (★, to put it at the top of the menu), the dates (**Today**, **Yesterday**, **Last working week**, this or last week or month, or none), **Folder** (only notes in folders matching it, written as the path is shown, e.g. *Clients > Prospects*; empty means every folder), **Keep the most recent** (only that many of the notes found, the most recently written; 0 keeps them all), **Skip selection** (run the first stage as soon as the notes are found, without stopping at **Notes considered**; **Cancel** it to change the notes, then **Run**), **If none, use the most recent day** (when nothing falls in the dates, it goes back to the latest day with notes, in the **Folder** if one is set, and says so), a model for every stage (or the library's answer model), and the stages. Each stage has a name, its output (**Checklist to pick from**, **Written output**, or **Run an Agent**: the chosen Agent is injected into the Ask box with the stage's prompt as the question and **Ask about in-scope notes** runs, so its items are staged under the answer for you to review, drop and send as usual), its prompt (typed, or **A note** from a library, read each time it runs), its own model if you like, whether to send the notes in scope, and **Skip and approve step** (the same tick as in the running panel). A Checklist stage sets how its list is grouped at first (client / subject, person or kind), and can **Route each item**: instead of a tick, each item gets **NFA** (no further action), **Agent** or **Message (via user)**, with the same three as buttons per group. Every item has a :material/edit_note: button to correct it in words (*the owner is Sam, due Friday*): the box grows as you type, and the correction goes on with the item for the later stages (and **Change this list**) to apply. In a prompt, `{week}` is the date window, `{selection}` the items you kept, `{system}` the items routed to Agent, `{message}` the items routed to Message (under a heading per person), `{context}` what you added beside the checklist, `{previous}` the last stage's output and `{earlier}` every earlier stage's written output by stage name. Written output from earlier stages (e.g. an email drafted at step 2) is added to a later stage's prompt as context by itself, unless the prompt places `{earlier}` or already takes it as `{previous}`. A stage whose prompt needs something an earlier stage gives (`{selection}`, `{system}` or `{message}` need a Checklist before it; `{previous}` needs a stage before it) shows a warning when it's moved above it. Reordering a workflow that's running closes its run, to run again in the new order. A stage whose `{system}` or `{message}` has nothing routed to it says so and can be skipped. **Add stage**, change the order (each stage's **Step** picker moves it straight to another step; the arrows move it one place), **New workflow**, **Duplicate** (a copy of the workflow and its stages, just under it and open to change, so you can build a variant without touching the original) and **Delete workflow** are there too.

**last-mtn-action** is an example of a workflow set up that way: Today, keep the most recent 1, falling back to the latest day with notes; **List the actions** (a Checklist of each action, owner and due date in that meeting, with your context beside it), with each action routed to NFA, Agent or Message; then **Create the tasks** (runs the SF_Tasks Agent on the Agent actions) and **Draft the messages** (an email with a heading per person and their Message actions, to **Copy** and send from your own mail app; nothing is sent for you). "Most recent" follows **Note dates**: the date written in the note (with its file's creation time to order notes on the same day), its creation date, or whichever is later.

**Note dates** (sidebar **Configure** dropdown) sets how date filters find a note: by a date written in it or its title (e.g. `2026-11-05`), by its file's creation date, or either (the default). It applies to the quick date buttons, a workflow's week and dates you type as filter terms (`2026-11-05`, `2026-11`, `2026`, `Week#45`).

The sidebar ends with **Corpus index** (the last scan or rebuild) and **Debug details**, both folded.

To switch quickly, **Select Library** lists **Single Corpus** first, with when the whole corpus was last indexed (*Single Corpus · indexed 26-11-08:08:33*, or *not indexed yet*), so you can tell whether it's up to date. The Filter box has a copy of **Select Library** above **Max results** (both routes), and after an all-library literal search **Matches by library** has a **Use Single Corpus** button. Hovering a dropdown's cut-off text shows all of it. Picking it is the same as choosing Single-Corpus mode, and picking a library switches back to that library. The first time, the whole corpus needs **Re-index (full rebuild)**.

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
python3 -m streamlit run UI-semantic-search.py --server.address localhost --server.port 8510
```

`start_app.sh` (and the Desktop launcher from `Installer.sh`) starts **Ollama if it is not already running**, then warms the **Answer with LLM** model configured in your corpus settings (most common `ollama_model` across libraries, usually `llama3.2:3b`). Embedding models are **not** pre-loaded at startup.

Re-run `./Installer.sh` if you move the project folder or need to refresh the Desktop launcher.

The app always runs at **`http://localhost:8510`**, so you can bookmark it. If it's already running, `start_app.sh` and the Desktop icon just open it instead of starting a second copy. Use another port by setting `LLM_NOTES_PORT` (e.g. `LLM_NOTES_PORT=8520 ./start_app.sh`).

### Start, stop and quit

The Desktop **LLM-Notes** icon (made by `Installer.sh`, or `./llm_notes_service.sh install`) runs the app in the background on `http://localhost:8510`, with no Terminal window:

- **Click it** to open LLM-Notes in your browser, starting it first if it isn't running (the first start can take a minute). It never starts at login; nothing runs until you click.
- **The first time,** macOS asks whether *LLM-Notes* may access files in your **Documents** folder (where the project lives). Click **Allow**. If you clicked Don't Allow, turn it on in **System Settings → Privacy & Security → Files and Folders → LLM-Notes**. Rebuilding the icon (`install`, or re-running `Installer.sh`) can make macOS ask again.
- **While it runs, its icon stays in the Dock.** Right-click it and choose **Quit** (or press ⌘Q with it in front) to stop LLM-Notes and free its memory. Clicking the icon again just opens the page.
- **Stop it from the project folder** with `./stop_app.sh`, whether it was started from the icon or from Terminal. The Dock icon then leaves by itself.
- Output goes to `~/Library/Logs/LLM-Notes/llm-notes.log`. It doesn't restart itself after a crash: click the icon again.

```bash
./llm_notes_service.sh install     # rebuild the Desktop icon (doesn't start it)
./llm_notes_service.sh open        # what the icon does
./stop_app.sh                      # stop it (same as ./llm_notes_service.sh stop)
./llm_notes_service.sh restart     # after updating the app (git pull)
./llm_notes_service.sh status      # running and answering?
./llm_notes_service.sh logs        # follow the log
```

Set `LLM_NOTES_PORT` to use another port.

**As its own app in the Dock.** With LLM-Notes running, open `http://localhost:8510` and, in Safari, choose **File → Add to Dock**; in Chrome, **⋮ → Cast, save and share → Install page as app**. It then opens in its own window with its own Dock icon, separate from your browser tabs. If LLM-Notes isn't running, the window shows an error: click the Desktop icon to start it. To quit, right-click the LLM-Notes icon in the Dock, not the web app's.

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

1. Set **Notes location** (under **Configure** in the sidebar) to your Apple Notes export folder (or use **Browse…**).
2. Select a **library** (corpus) in the left sidebar.
2. Open **Advanced — Indexing & Embeddings** if you want to change the embedding model or relevance cutoff.
3. Click **Re-index (full rebuild)** for a new corpus, or **Rescan (incremental)** to pick up new, edited and deleted notes.

The sidebar groups these under **This corpus**. Beside **⚡ Rescan** is **⚡ Rescan all corpora**, which rescans every visible corpus in turn, also incrementally (see [Rescan vs Re-index](#rescan-vs-re-index)).

Progress appears in the sidebar: a live progress bar during the scan, then a persistent summary (docs, blocks, vectors, model used).

**First index** does not require the “I understand rebuild” checkbox. That acknowledgement only appears when changing the embedding engine on an existing index.

### Search

1. Choose your library.
2. Optionally add filter terms as capsules (type a term, press Enter):
   - **Must include (AND)** — every term must appear in the note.
   - **Any of (OR)** — at least one term must appear.
   - **Exclude (NOT)** — notes with any of these terms are dropped.

   Terms match literally and case-insensitively anywhere in the note, its file name or the folders it's in (`2026-10` matches `2026-10-19`; `Clients-Prospects` matches every note in a folder of that name, and `Clients/Prospects` or `Clients > Prospects` match nested folders). A note whose file name appears in two folders of a library can't be placed, so only its text and name are matched. With a question, semantic search ranks passages only within the matching notes. Filters are saved between sessions.

   Each field sits in its own small box: its heading, quick dates and quick search terms (rounded chips), then its terms. **Quick dates** sit on the heading line of **Must include**, **Any of** and **Exclude**, and put date terms into that field. Clicks add up until you search, so **Today** then **-1 day** in **Any of** finds notes from either day. After a search, the next click first takes out the dates that were in that field for the search (and dates restored when the app starts), whether a button added them or you typed them (`2026-10-25`, `2026-10`, `2026`, `2026/10/25`, `25/10/2026`, `Week#43`…). Other terms, and the other two fields, stay as they are:

   | Button | Adds to Must include (on Sunday 25 October 2026) | Adds to Any of / Exclude |
   |---|---|---|
   | **Today** | `2026-10-25` | the same |
   | **-1 day** | `2026-10-24` | the same |
   | **This wk** | `2026` and `Week#43` | the 7 dates `2026-10-19` … `2026-10-25` |
   | **-1 wk** | `2026` and `Week#42` | the 7 dates `2026-10-12` … `2026-10-18` |
   | **This mnth** | `2026-10` | the same |
   | **-1 mnth** | `2026-09` | the same |

   Days match notes that contain the date as `YYYY-MM-DD`, and months every date in that month. In **Must include**, weeks match notes headed `Week#NN` in that year, using ISO weeks (Monday to Sunday; 29 Dec 2025 is 2026 `Week#1`). The year is its own term rather than a month, so a week that spans two months still matches. Because terms match anywhere, `Week#4` also matches `Week#40`–`Week#49`, which only matters in weeks 1–9. In **Any of** and **Exclude** a bare year would match, or exclude, the whole year, so weeks are the seven days instead. A bare four-digit number counts as a year, so a button would take a leftover one out too. Hover a button to see exactly what it puts in.

   **Quick search terms** sit on a second line under the quick dates, on each of the three fields: up to 5 terms you filter by often, such as a customer or a folder. A click adds the term to that field (once). Set them on the prompt library page under **Quick search filters** (folded by default); the ✏️ link beside the buttons (**Set quick search terms** until you have some) opens that section in a new tab. They're saved as you type and show on the next rerun of the search page.
3. Enter your question in **What would you like to ask?**
4. Click **Search**.

The search area is split into a **Filter** box (left) and an **Ask** box (right).

**Search route.** The switch at the top of the Filter box picks one of two routes into your notes; the Filter box shows only the settings for that route, and the Ask box's button follows it:

| Route | What **Search** does | Settings shown |
|-------|----------------------|----------------|
| **Semantic (meaning)** | Ranks passages by meaning using the embeddings. Filter terms narrow which notes it looks in. | **Find notes about** (below), the filter folded into **Narrow by terms** (its title says how many terms are set; when terms are set it opens, with any **Any of** and **Exclude** fields holding terms, as you start a new search: a new question or **Find notes about** text, or a **Search** or **Find notes** click, and stays as you left it in between), **Distance cutoff** (how close a match must be) and, after a search, the match bar. |
| **Literal (exact text)** — button reads **Ask (filtered only)** | Returns the notes that match the filter terms, in full. Much faster; the question is only used for the LLM answer. | **Search all libraries**, **🔎 Filter search only**, **🔄 Rescan → Filter search**. |

**In scope: gather across searches, then ask.** Each route has its colour: **Semantic** blue, **Literal** orange. The route switch shows both (the chosen one a few shades deeper) and the Filter box takes the chosen route's tint.

- **Gather.** Every search adds the notes it finds to **In scope**, a list under the Filter box grouped by folder: **Search** / **Ask (filtered only)** with an answer, **Find notes** (Semantic: type what the notes are about, then **🔎 Find notes** at the bottom of the Filter box) and **Filter search only** (Literal). Each note is tinted with the colour of the search that found it last, so a note found by both routes takes the later one's colour. Searches always look through the whole library, so you can, say, search one company's conversations, then another's, and have both in scope.
- **Down-select.** Tick **🚫** to drop a note, or click **🎯** to keep just the ones you pick.
- On the Literal route, **Ask (filtered only)** shows how many notes it would send (*· 12 notes*, or *· 15 of 40 notes* when **Max results** limits it). **Ask about in-scope notes** is only shown when the in-scope notes you kept differ from those.
- **Ask about in-scope notes** (in the Ask box, with the count) answers your question (or the chosen task) from every in-scope note you kept, e.g. *list the actions*. Each note gets an equal share of the **Notes budget** (12,000 characters of notes by default), so all of them reach it; the answer says when notes were trimmed to fit.
- **Inputs and buttons follow their section's colour:** fields get a light tint of it with a mid-tone border, buttons a deeper tint that deepens on hover (Filter box blue or orange by route, Ask box and answer green, Agent box blue). Text keeps the theme's colour, so it stays readable in light and dark mode.
- **The LLM side is green.** The Ask box, where you put the question together, and the answer below (with its sources and discussion) have a light green tint; an Agent's workflow (steps, staging, sent actions) sits in a harder blue inside it. Once an answer is showing, **In scope** folds into one full-width bar above it (*In scope · N notes, K kept*): open it to look at or change the notes.
- **Clear results** (beside **Clear Filter**) empties the list and the last results; filter terms and exclusions stay. **Clear Filter** also empties the list. **Clear all** (rightmost, outlined in red) starts the page afresh: filter, exclusions, results and discussion, the question, **Find notes about**, the **Oneshot search**, the **Actioned** filter (back to *All notes*), any injected task prompt and the **☁️ Salesforce** reference (switched off). Settings such as the route, **Max results** and the model stay.

The route is saved between sessions. **🔎 Filter search only** (Literal route) lists the matching notes under the Filter box without an LLM answer, keeping your question for later. Notes are grouped under their folders as a tree, sorted by name, each 📁 folder with the number of notes under it (sub-folders included); a folder holding no notes of its own and one sub-folder is merged with it (e.g. `Customers > Acme`), and notes whose folder isn't known come last. Each folder has its own **🚫** and **🎯**, acting on every note in it and its sub-folders: e.g. 🎯 on *Customers* keeps all customer notes and excludes the rest of the list. With more than 50 notes listed, the folders fold up: **▸** opens one to show the notes directly in it, so a listing of hundreds of notes stays quick. Tick **🚫** beside a listed note to exclude it from the next **Search** (semantic ranking and LLM answer); excluded notes stay in the filter list, crossed out, and **Clear** (beside *🚫 N excluded*) resets them for the session. **Max results** (title beside its box), the filter in words (*Filter: 2026-10 AND STC*), how many notes are excluded and **Clear** share one line, with Clear rightmost. **Any of (OR)** and **Exclude (NOT)** start folded to keep the Filter box short; a fold's title shows its terms when it has some (*Exclude (NOT) · 2: draft, old*), and opening it gives the same quick dates and quick terms as **Must include (AND)**. Once you've added a term in one, it stays open (as does **Narrow by terms**) until **Clear Filter**. A full **Search** shows the answer, sources and discussion full-width below both boxes.

On the Literal route, a note shows when the text index was last rescanned (for this library, or every visible library when **Search all libraries** is ticked) and whether auto-rescan is on. **🔄 Rescan → Filter search** picks up new, edited and deleted notes first, then runs **Filter search only**. It is text only, so the filter sees the changes straight away; semantic search sees them after **⚡ Rescan**.

The **Literal** route skips vector search and returns the matching notes in full (the question is optional and only used for the LLM answer). Add **Search all libraries** to run that literal search across every visible library: a *Matches by library* panel shows note counts per library, with a **Use library** button to switch to one for a semantic search. Only the selected library's notes are listed (and gathered in scope); to list notes from every library, use **Single Corpus**. When **Max results** keeps fewer of this library's notes than match, it says so (*Showing 1 of 5 matching notes: Max results is 1*) with a **Show all 5** button that raises Max results to fit and lists them again. No embeddings are used, so nothing needs re-indexing.

Results appear under **Sources** as one dropdown per document (documents with several matches are grouped). Each dropdown has two tabs:

- **Matched passages** — the retrieved chunks with their rank and distance.
- **Full document** — the whole file embedded in the page (markdown is rendered; PDF/DOCX/PPTX text is rebuilt from the indexed blocks), plus a download button for the original.

### Answer with LLM

The Ask box reads top to bottom: a title line with **Ask**, **Answer with LLM** and the injected prompt's **links** (if it has any, e.g. Salesforce tasks) → your question (one line high; **Bigger**, on the line above it, makes it taller) → **Prompt library** → **Ollama model details** → **Search** → progress. Search settings are in the Filter box: **Max results** under **Exclude (NOT)**, then the route's own settings at the bottom (**Distance cutoff** on the Semantic route; **Search all libraries**, **Filter search only** and **Rescan → Filter search** on the Literal route).

- **Answer with LLM** is enabled by default.
- **Prompt library** — see [Prompt library](#prompt-library) below.
- **Ollama model details** is folded by default; a line under it shows the model, max tokens (with how long that would take at the model's last measured speed) and temperature in use. Inside: **Ollama model**, **Max tokens**, **Notes budget** and **Temperature** (saved per corpus as soon as you move them).
- **Notes budget** is how much note text **Ask** sends the LLM (6,000 to 160,000 characters; 12,000 by default; about 3 characters a token). Raise it when notes are trimmed. It's capped to what the answer model's context window holds beside **Max tokens** (Ollama's window here goes up to 64k tokens), and the line under it says so. More notes text means slower answers and more memory. The Chat page's **Context window** is separate: it only applies to Chat.
- **Max tokens** goes up to 50,000 (≈37,500 words). Its stops include book lengths for scale — *The Very Hungry Caterpillar*, *Where the Wild Things Are*, *The Tale of Peter Rabbit*, Dr. Seuss's *The Cat in the Hat*, *The Little Prince*, *Winnie-the-Pooh*, *The Old Man and the Sea*, *Animal Farm* and *The Lion, the Witch and the Wardrobe* — and the label and the line under the slider say how long the current setting is (word counts are approximate; English runs at about 0.75 words per token). *The Hobbit* (≈127,000 tokens) and *War and Peace* (≈783,000) are beyond the limit and shown for scale. The limit includes a reasoning model's thinking. Long answers take time (at 20 tokens/s, 50,000 tokens is about 40 minutes) and use more memory: Ollama's context window grows to fit the prompt plus the answer (8k, 16k, 32k or 64k tokens).
- **Max results** (a slider in the Filter box, under **Exclude (NOT)**) caps how many passages (Semantic) or notes (Literal) a search returns, from 1 to 200: 30 to start. Each route keeps its own value. On the Literal route, **Filter search only** lists up to the full number, while **Ask (filtered only)** answers from at most 50 notes, since the LLM only reads the **Notes budget** of note text (12,000 characters by default).
- **Distance cutoff** (at the bottom of the Filter box, Semantic route only; same setting as *Relevance cutoff* in the left panel, saved per corpus) decides which passages count as semantic matches, so it applies with or without an LLM answer. After a semantic search, a bar under it shows where passages were found — green is within the cutoff, amber is where to raise it — with a **Set cutoff to …** button to include the closest match.
- **Progress** shows right under **Search**: a spinner with the elapsed time while searching and answering, and beside it the generation speed (tokens per second and tokens so far), refreshed every 3 seconds. Answers stream from Ollama, so there's no overall time limit. The answer caption ends with the final tokens per second. Follow-up questions (and Narrate / Re-run as text) show the same spinner and speed while you wait, and each follow-up reply's model label ends with its tokens per second.
- **Speed score.** After each answer the model's speed is saved (`llm_speed` in `app_settings.json`, per model) and shown under **Search** between runs, e.g. *Last run with `qwen3:30b` (25 Oct 10:42): 12.4 tokens/s · 10,000 tokens ≈ 13 min 26 s*, for whichever model is selected.
- Search runs first; Ollama summarises the retrieved chunks—it does not replace retrieval.
- **Ask a follow-up and the Agent box stay in view.** In the side panel, the **Ask a follow-up** box and, when an Agent is attached, the Agent box under it travel down the page as you scroll a long answer, so you can answer its questions and stage without scrolling back. If the two together are taller than the window, they scroll within themselves. The transcript above them scrolls away as usual.
- **Follow-up model** sits in the title line of the **Ask a follow-up** box, beside **Expand**, with the chosen model's last measured speed under it (*10,000 tokens ≈ …*), so you can pick one you're willing to wait for before sending. It can differ from the answer model and is saved per corpus; the **Pop out** window uses the same one.
- While the LLM is working (an answer, a Discussion follow-up or an Agent step), an orange **Cancel Query** button floats at the bottom of the window wherever you've scrolled. It stops the request within about half a second, even while the model is still reading the notes, and Ollama stops generating. Change the task, prompts or filter and run it again.
- **Folder paths.** Each result gets its note's folder path from where the file sits under the notes folder, e.g. *Clients > Prospects > Acme*. It's shown on the note in **Sources** and given to the LLM with every passage (`[Source 2: Kickoff-2026-10-20.md, in Clients > Prospects > Acme]`), so the model knows what each note is about even for passages that don't include its title. No rebuild is needed. When two notes in a library share a file name, the path is left out rather than guessed. Local Apple Notes Export can also put the path in each note's title line (its *Folder path in titles* option), which puts it in the indexed text for search, after a rescan and re-embed.

### Dictation

Speak instead of typing in the **Ask** box (the mic in the question's title line), the **Ask a follow-up** box (the mic in its title line) and **Chat with LLM** (**Dictate a message**, at the top of the sidebar, where **Dictate to Chat** is on the other pages). It's hands-free: click **Dictate** and speak; when you pause for **3 seconds** it's sent by itself (a countdown shows *Sending in 2 s…*; click the mic again to send sooner). If it hears nothing for 15 seconds it stops without sending. Your words are transcribed **on this Mac** with Whisper (`openai/whisper-base`, run through the transformers library the app already uses), so the audio never leaves your machine.

- **Dictate hotkey** (on by default, the backtick key; mic labels show it as *(hotkey [ ` ])*): press it to start dictating, and again to send at once (or just pause). In Chat it dictates a message to the chat. On any other page (search, Prompt library, Configure, Actions, Reference) a red bar appears at the top while it listens: *Chatting by hotkey with [model]*, with a **model** dropdown (kept for next time; changing it doesn't stop the listening), an **About me** toggle (on by default, your choice kept for next time: the new chat sends About me from the Prompt library), **Cancel** (stops listening, sends nothing, you stay put) and **Open in a new tab** (a new tab gets the message when you pause; this one stays where it is). When you pause, you're taken to a new chat, where it's transcribed (with the model, **About me** and **Cancel** again) and sent straight away. **Back to …** in the Chat sidebar returns you to the page you were on, as you left it. The sidebar's **Dictate to Chat** mic does the same with a click. While the reply is written, the floating **Stop** has the same model dropdown and **About me** toggle: changing either stops that reply and asks again. Once it's answered, changing **About me** above the chat asks the last message again, with or without it (the reply is replaced). The search page's own mics still dictate into Ask and the follow-up box. While you're typing in a box the key types as usual, so click outside the box first. Turn the hotkey on or off with the **Dictate hotkey** toggle (search sidebar, Chat sidebar or Configure), and choose the key under **Configure > Dictation and Chat**, where an optional **Second hotkey** does the same (e.g. for a second keyboard layout; labels then show both, *[ ` ] / [ § ]*): `` ` ``, §, ~, \\, |, F2, F4, F8 or F9. On some keyboards ` and ~ are accent keys that wait for the next letter; pick another key if pressing it does nothing.
- **Cancel transcription:** every mic shows a floating **Cancel** while its recording is transcribed; it stops the wait and sends nothing.
- While a dictated question is answered, the floating button reads **Stop and edit**: it stops the reply and leaves your text in its box (a chat message is taken back out of the chat into the chat box) to fix and send again.
- The first time, the speech model is downloaded from Hugging Face (about 290 MB) and loaded, which takes about a minute; after that it's loaded once per app start (about 25 seconds) and works offline. A short question then takes a few seconds to transcribe.
- The mic listens in the browser for the pause, so it can send without a stop button; a red level bar shows it hearing you.
- The browser asks once for microphone access.
- **Read aloud:** the answer to a dictated question is read out in the Mac's own voice (`say`, using the voice set in System Settings > Accessibility > Spoken Content), also made on this Mac. It plays once, by itself, from a player under the answer that can pause, rewind or replay it; Markdown symbols, link addresses and code blocks are left out. Typed questions aren't read out. So the answer is written to be heard, the **Read aloud (dictation)** prompt in the Prompt library ("Prepare the answer in a way that Siri can read it as a dictation.") is added to what's sent with a dictated question; edit its wording there, or delete it to stop it being added. Narrate (below) uses it too.
- **Narrate, Read as is, Re-run as text:** under any text answer (the Ask answer, a follow-up reply or a Chat reply), **Narrate** asks the same question again with the **Read aloud (dictation)** prompt and reads that answer out, and **Read as is** reads the answer out as it stands. Under a narrated answer (one to a dictated question), its player (or **Play**) sits with **Re-run as text**, which asks again without the prompt. The new version appears indented under the original, which stays; doing it again replaces that version. Only the conversation up to that answer is sent, so later messages don't sway it, and later messages never see the new version. It's kept with its answer: in the chat's file for a saved chat, and only in the session for Ask, follow-ups and in-confidence chats. The audio is never saved. Not offered for Agent runs, whose replies lead to drafts, or in the pop-out discussion window.
- **Try another model:** beside those buttons, pick any other installed Ollama model to answer the same thing. It's sent exactly what the original model had: the same prompt, notes and conversation up to that answer (nothing after; a narrated answer stays narrated), with the current Max tokens and Temperature. Its answer appears indented under the original, labelled *Another model* with its name and speed. Try several models to compare them; trying the same model again replaces its version. These are kept like the narrated versions above, and the conversation carries on with the original model.
- **Edit question:** like editing a notebook cell, change the question behind any reply (the Ask answer, a follow-up or a Chat reply) and run it again with any installed model. Only the conversation up to that point is sent, with your edited question in place of the original; the Ask notes aren't searched again. The answer appears indented under the original as *Edited question*, showing what was asked; the original and everything after it stay as they were. Running the same edit with the same model again replaces that version.
- **Narrate on any output:** output with no question to re-ask (the indented versions, Agent replies and the Big screen reply) has its own **Narrate**, which has the model that wrote it rewrite that text for listening with the **Read aloud (dictation)** prompt, then reads it out. The rewrite appears just under it as *Rewritten for listening*.
- Turn read-aloud off with **Read answers aloud when I dictate**, under **Configure** in the sidebar (or in the Chat sidebar).

### Chat with LLM

**Chat with LLM**, at the very top of the left sidebar, opens a classic chat with any model installed in Ollama, in the same tab (**Back to search**, or the 🧠, returns to the search page with everything as you left it).

- **Chats are saved** once they have a message, as `llmware_data/chats/<id>.json` on your machine only (`llmware_data/` is in `.gitignore`, so chats are never committed). They're listed in the sidebar, most recent first; click one to carry on, or click the **×** beside it to delete it straight away (no confirmation). **New chat** starts afresh. A chat is titled from its first message; **Rename** and **Delete** sit beside the title.
- **Open on Chat** (toggle in the Chat sidebar, or under **Configure > Dictation and Chat**): the app opens on Chat with LLM instead of the search page, so you can use it as your main page. **Back to search** still goes to the search page.
- **Notify me when a task finishes while I'm away** (**Configure > Dictation and Chat**, on by default): when LLM work finishes while you're on another tab or app (an answer or Oneshot search, a Note-Workflow stage, an Agent's answer or **Execute**, a Chat reply, a dictation), a macOS notification says what finished and where, e.g. *Note-Workflow · List the actions is ready to check*. It holds nothing from your notes. Click it to bring the app's tab back to the front. Work that takes under 8 seconds, or that finishes while you're looking at the tab, doesn't notify. The browser asks to allow notifications once, on your next click on the page; if none show, allow them for `localhost` in the browser's site settings, and for the browser in **System Settings > Notifications**.
- **Notes questions go to search** (Chat sidebar toggle *Send dictated notes questions to search*, on by default): a **dictated** message (mic or hotkey) that needs your notes runs as a search instead. Typed Chat messages are never checked and always stay in Chat; for your notes, type into **Oneshot search** on the search page. Missed one, or typed it in Chat? **Push to notes as question** (beside **Edit** under each of your messages) runs that message as a Oneshot search on the search page, with no check first; **Back to chat** returns to the chat, which notes the push. Guessed wrong? **Send to non-note search** (in the pop-up, and on the blue bar) stops the search and sends the message to Chat as an ordinary message: back into the chat it came from (replacing the chat's note of the move), or a new chat for a Oneshot search, still read aloud if it was dictated. Clear requests always count (*search my notes…*, *find notes…*, *my notes*, *where did I mention…*, *what did we agree…*, *what do I know about…*, *search for…*, *look up…*, *RAG*); carrying on the chat (*make that shorter*, *translate this*) never does; everything else gets a quick yes/no check by the **RAG selector model**, told what counts (your meetings, calls, colleagues, customers, deadlines, what you wrote or said) and what doesn't (general knowledge, how-to, writing new text) (sidebar **Configure** dropdown; *Same as the chat's model* by default, or a small fast one such as `llama3.2:3b`). General questions, writing and explanations stay in Chat. The model splits the message into what to **find** and the **intent**: *search my notes for mentions of the side car and find details about missed deadlines* finds `side car` OR `side-car` OR `sidecar`, then asks *find details about missed deadlines* of those notes. It's told how literal matching works (the topic's own words, one per term, alternatives only when you give them, never words like *notes* or *projects*), and the app also strips such words, keeps intent words out of the filter and exclusions to what you said to leave out, tries every spelling of a word (*heat pump*, *heatpump*, *heat-pump*), uses dates only when you name a period, and a library only if you name it (otherwise **Single Corpus**, or your current library if the whole corpus isn't indexed yet). Then it runs as a **Oneshot search**: your words appear in the Oneshot box on the search page, exactly as if you'd typed them there, and it runs step by step in a **Searching your notes** pop-up (the blue bar says *Oneshot from Chat*): find the notes by their words (or by meaning if the message is a broad theme); if none match, try by meaning; keep every note found in scope; then **Ask about in-scope notes** with your intent. Nothing found? The pop-up says so, with your message to edit and **Search again**, or **Say it again**. The answer is written in the background: click anywhere on the page and the pop-up shrinks to a chip (*Open*, *Cancel*) while it carries on; only **Cancel** stops it. It warns when notes are trimmed to fit the LLM's notes budget, when many notes get only a small share each, and when the answer hits **Max tokens** (or a thinking model spends them all thinking). A blue bar shows what was understood, with a library dropdown to run it all again elsewhere and **Back to chat**; the chat keeps a note of the move. For a dictated message the pop-up has **Read the answer to me** (on by default, your choice remembered): the answer is written as usual, then rewritten for listening with the **Read aloud (dictation)** prompt as a second step, shown under it as *Rewritten for listening* and read out. Never in an in-confidence chat. The two prompts behind it, **RAG check** and **RAG search**, are in the Prompt library under **RAG search from Chat**: change them, or **Reset to default**.
- **Edit** under each message you send: while its reply is still being written, it stops that reply and lets you fix the message in place; **Send** asks again. Once the reply is in, your edited message is asked as a new prompt (pick any model) and its answer goes indented under the reply, labelled *Edited prompt*, leaving the original as it was.
- **In-confidence chat** (sidebar, under **New chat**) starts a chat that isn't saved or logged. What you type or dictate and the replies stay only in that browser tab's session: no chat file, no entry in the chat list, no model speed recorded, and a read-aloud answer isn't cached. A lock banner shows you're in one. It's forgotten when you click **End and forget**, start or open another chat, leave Chat or close the tab. The app and Ollama don't log what's sent to the model (unless Ollama itself is run with `OLLAMA_DEBUG`).
- **Model** (per chat) can be changed mid-chat to have another model carry on; new chats start with the last one used. The line under it shows the model's last measured speed.
- **About me** sends your About me from the Prompt library as background. **Prompt** sends a prompt from your library as standing instructions with every message.
- **Attach notes in scope (N)** attaches a copy of the notes kept in scope on the search page (gather them there first with a search, Find notes or Filter search only), up to 24,000 characters shared between them. They're sent with every message in that chat; **Notes attached** lists them, and a click removes them.
- **Model options** (per chat):
  - **Max tokens** (4,096 to start, room for a reasoning model to think) and **Temperature**.
  - **Context window**: how much of the chat (and attached notes) the model sees at once. **Auto** sizes it to the chat, capped at the model's own limit (e.g. 131,072 for `llama3.2:3b`), but each step up reloads the model, so that reply starts slower. A fixed size avoids the reloads; a smaller one uses less memory.
  - **Thinking**: *Model default*, *On* or *Off*, for models that can think (e.g. Qwen). *Off* is much faster. Greyed out for models that can't.
  - **Top P**, **Top K**, **Repeat penalty** and **Seed**: empty means the model's own default (from its Modelfile), which usually suits it best. A seed makes replies repeatable, to compare models or options.
  - **Reset to defaults** puts the context window, thinking and sampling back.
- Replies stream in as they're written. A reasoning model's thinking streams into a folded **Thinking** section above its answer (*Thought for 12s*), open it to read. Under each reply, **Copy** and a stats line: model, **tokens per second**, tokens written, time taken and how much of the context window was used. Hover it for the breakdown: writing speed, prompt reading time and speed, model load time, time to the first token and the window size.
- **Context meter**: under the model line, how full the context window was at the last reply (*Context: 3,120 of 8,192 tokens (38%)*). Past 85% it warns that the model is about to stop seeing the oldest messages: pick a bigger window or start a new chat.
- The orange **Cancel Query** stops a reply. Your message is kept; **Get a reply** asks again, e.g. after switching to a faster model.

### Prompt library

Open it with the **Prompt library** button in the left panel (hover it for a summary of what is switched on), or the **↗** beside the title of the Ask panel's Prompt library box. It opens in a new tab (`?page=prompts`) and saves to `llmware_data/prompt_library.json` (the corpus prompt stays in `corpus_settings.json`). Top to bottom: About me, corpus prompt, quick search filters, reference documents, task prompts and Agent prompts. Links, the outbox and the title marker are on the [Configure](#configure) page.

| Section | What it does |
|---------|--------------|
| **About me** | One free-text box sent with every LLM request in every corpus — your name, where you're based, accessibility needs, how you like answers. Untick **Include in LLM requests** to switch it off. Today's date is always added. **Short name** (e.g. `JDOE`) signs the updates Agents make, such as a Next Step (*2026-10-25-JDOE: …*); it isn't sent to the LLM. |
| **Corpus prompt** | Pick a corpus and describe the role you wrote its notes in (e.g. business owner, salesperson). Tick **Apply corpus prompt** to send it for that corpus only. |
| **Quick search filters** | Folded. Up to 5 terms you filter by often, shown as buttons under the quick dates on each filter field (see [Search](#search)). |
| **Reference documents** | Folded. Notes the LLM gets as background, e.g. a process overview. Each document has a **Short title** (e.g. *PoC types*) and one or more notes. **Choose notes** switches this tab to the search page in pick mode, laid out in one column: the Filter box, then the document you're filling (its notes so far), then the list to add from (there's no Ask box). Gather notes with the Filter box (either route) and **Add** them in the list (or take them out). **✓ Done — back to Reference documents**, above the list and again after its last note, brings you back here with the document's notes filled in. Each listed note has **Open** beside **Add**: it shows the note in a popup, so the list, filters and selections stay as they are, and puts `?doc=<Apple Notes ID>` in the address (refreshing or sharing the link reopens it; the browser's Back button closes it). In the popup, **Add to <document>** becomes **Apply and return** (saves it and closes the popup) and **Remove from list** (undoes the add; the popup stays open); a note already in the document offers **Remove from list** and **Return**. **Open** on a document (here and in pick mode) shows all its notes in a popup, as the LLM gets them. A note is kept by its **Apple Notes ID** (from the Notes exporter's state file, shown as a short code), so renaming or moving the note keeps the link; notes not exported with the exporter are kept by path. Up to 20,000 characters of each document are sent. Attach documents to prompts (Agents or not) with **Reference documents** on each card: they're sent whenever that prompt is used. |
| **Task prompts** | Reusable instructions such as "Write a meeting summary and next steps as an email". Each has a name, text, **Global**, **Starred** and **Agent** checkboxes, a **Model** box, tags (for finding prompts) and a delete button. Edits save as you go, with a *Saved* time under the text; leaving a box that saved a change flashes it green with a **✓ Saved** tag (everywhere a box saves when you leave it). Tick **Global** to send a prompt with every request in every corpus, like About me — for standing instructions. Tick **Agent** to make it a workflow that ends by sending to the outbox (see below); Agent prompts have a blue border. **Model** picks the Ollama model used when you choose the prompt (see below). Above the list: full-text search (covering both sections), **Clear search** and **Common keywords** (tags from your most recently used prompts) to filter with one click. |
| **Task prompts / Agent prompts** | Two collapsible sections, open until you close them (remembered): prompts with **Agent** ticked are under **Agent prompts** (blue outline and heading, its inputs and buttons shaded blue), the rest under **Task prompts**. Ticking or unticking **Agent** moves a prompt at once; Agents show as *Agent:Name* and are never renamed, and each Agent card has a blue outline and a light blue hue, folded or open, so it reads as agentic at a glance. **↑ / ↓** beside each card move it within its section (the order is used everywhere prompts are listed). Each prompt is a collapsible card: its header shows the name (with *· Global* and *· ★* markers) and opens it; cards start closed, each section has **Expand all** / **Collapse all**, and open cards are remembered (in the app's settings). An Agent's **Edit prompt workflow** link opens its card. |
| **Add a prompt** | Name, text, **Global**, **Starred**, **Agent**, the Agent's output, **Model**, tags, **Reference documents** (any from the document library, sent whenever the prompt is used) and **☁️ Send Salesforce records**, then **Add prompt**. Add an Agent's steps and Action Command on its card afterwards. |

**Model per prompt.** Leave it on *Model: as selected* to use whatever is chosen in the Ask panel. Pick a model (any you have pulled into Ollama, e.g. a Qwen model) and choosing the prompt selects it as the **Ollama model** for the answer and as the **Follow-up model** for the discussion. You can still change either one in the Ask panel; that overrides it for the run. A model picked by a prompt doesn't become the corpus's saved default.

**Agent prompts** are the only prompts that send anything to the outbox. Each is a workflow you refine as you use it:

| Part | What it is |
|------|------------|
| **Step 1 · Collection** | The prompt text. It runs with the search: what to find in the notes, list (numbered, so you can pick by number) and ask about. |
| **Steps 2, 3, …** | Optional. Each has a short name and a prompt, e.g. *Select tasks* ("List the tasks again and ask which to send") or *Set due dates*. They become one-click buttons in the Discussion panel. **Add step**, reorder with the arrows, remove with ✕. |
| **Output** | What the Agent sends: **Note**, **Tasks** or **Email** (the box next to **Agent**). |
| **Action Command** | The last step, e.g. *Send to tasks in Salesforce*. It's the instruction used to stage the result. |
| **Actioned mark** | Tracks which notes this Agent has actioned (see [Actions log](#actions-log)). Type it into a note in Apple Notes to log something done by hand. Empty means `[<Agent name>]`, e.g. `[SF_Tasks]`. |

**Agent Semantic Fixes** (below the Agent prompts, blue outline) are facts that repair what Agents send, e.g. *Joseph Oliver is known as J-O*. Each is a large text box, saved as you type, with the date it was added (and last edited); ✕ removes one. They're used only by **Re-assess** on the Actions page (see [Fixing an item with context](#fixing-an-item-with-context)), not when Agents stage records. They're kept in `prompt_library.json`, so they're included in a settings export.

A prompt can be Global or Agent, not both. Prompt libraries saved before Agent workflows existed are upgraded on load: prompts already set to draft a note, tasks or an email become Agents with no middle steps and a default Action Command (*Create these as tasks*, *Save this as a note* or *Draft this as an email*).

Four example task prompts are added the first time; edit or delete them freely.

**In the Ask panel** (shown when **Answer with LLM** is on), the compact **Prompt library** box sits above the model settings and has:

- **About me** and **Corpus prompt** — toggles on the box's title line. About me is on for each new session; switch it off to leave it out of this run. Corpus prompt is this corpus's **Apply corpus prompt** setting (saved, the same as on the Prompt library page), greyed out until the corpus has a corpus prompt written; hover it to read the prompt.
- **Always** — each global prompt, ticked. Untick one to leave it out of this run.
- **Task** — click a prompt to attach it (click it again, or **✕**, to remove it). Agents add up: click **Agent:SF_Next_Step** and **Agent:SF_Tasks** to do both in one action (see [Several Agents at once](#several-agents-at-once)); a plain prompt replaces whatever was attached, and an Agent replaces a plain prompt; your question still decides which notes are found, and the task tells the LLM what to write from them. The task also applies to follow-up questions. On the **Literal** search route, a task can produce an answer even without a question.
- **Recent** — titles of the five task prompts most recently used for an answer, newest first. **Starred** — starred prompts not shown in Recent, most recently used first, so favourites fill the gap. **Agent Starred** — the same for starred Agent prompts, on their own row under Starred. Each row shows only the titles that fit on one line after its label (a title that would overflow is left out; find it with **Find a prompt**). Hover a title to see the full prompt.
- **Find a prompt** — search every task prompt by name or tag.
- **Agent prompts** read *Agent:Name* (e.g. *Agent:SF_Tasks*) and are blue: their chips on the Recent, Starred and Agent Starred rows are shaded blue (deeper when chosen), and they have a blue border on those buttons, on the chosen **Task**, and on the prompt token under the question box. **Find a prompt** shows the same *Agent:* names.

- **Reference** — **☁️ Salesforce** (see *Salesforce context* below), then reference documents for this question: up to five recently used ones as buttons (titles that wouldn't fit on the line are left out), and **Find a reference document** for the rest. **Sending:** lists every document that goes with the request, including those the chosen prompts bring (named in brackets); click one to read it in a new tab.

Each request is sent in this order: today's date → About me → corpus prompt → global prompts → reference documents → notes and question → task. The answer caption lists the prompts that were used.

Requires Ollama running locally (`localhost:11434`).

### Configure

Open it with **Configure page** inside the **Configure** dropdown in the left sidebar's Library Controls (a new tab, `?page=configure`). It holds the settings for what happens to an Agent's drafts:

| Section | What it does |
|---------|--------------|
| **Task executor settings** | First, the **⚙️ Task executor**: the service that processes the outbox (e.g. the Salesforce Connector app at `http://localhost:8503`). It's defined in its own row, can be cleared but not removed, is listed first wherever links are, and is shown at the top of the **Actions** page with its 🟢/🔴 status. Then **🔗 Other links**: short-titled links (e.g. *SF Tasks*, *Connector*) that you attach to prompts with each card's **Links** selector. While a prompt is injected, its links show as hyperlinks between the question and the prompt token, opening in a new tab. Links to this computer (localhost) show 🟢 when the service answers and 🔴 when it doesn't (checked at most every 15 seconds). **Best guess search:** put `{best_guess}` in an address (e.g. a Salesforce search URL) and it's replaced with the app's best guess at what you're looking for, shown in the link text too (*Find in SF: Acme*): the Agent's staged or sent *Related to* names, else the first **Must include** filter term, else capitalised names in your question. Stored in `prompt_library.json`. |
| **Outbox** | The folder that sent drafts are written to, and how many packages are waiting, processed and failed. See [Agents and the outbox](#agents-and-the-outbox). |
| **Task title marker** | Text added **before** or **after** the title of everything an Agent stages (task subjects, note titles, email subjects), e.g. `RAG_TEST:` or `[AI]`, so those records are easy to find later. Applied when the Agent stages, so you see it in the review box; titles that already contain it are left alone. Empty means off. |

#### Export and import settings

The last section of **Configure**. **Download settings** saves one JSON file (`rag-workspace-settings-<date>.json`) with:

- the notes app: About me, every prompt (with Agent steps and Action Commands), task executor settings and other links, quick search terms, the title marker, each corpus's settings (embedding model, cutoff, LLM model and tuning, corpus prompt) and the app's settings (notes folder, corpus mode, outbox folder, auto-rescan, GPU, hidden corpora, Max results, search route);
- the Salesforce connector's `config.toml` and the Notes exporter's `config.json` (their paths are shown and can be changed; untick either to leave it out).

**Nothing from your notes:** no note text, indexes or outbox content, no reference documents (prompts lose their links to them), no excluded notes, filter terms, last question or speed stats, and none of the exporter's single-note include/exclude rules (they name notes).

To restore, e.g. on a new Mac: put the notes back where they were (export them again with the Notes exporter), choose the file under **Import**, tick the parts to restore and click **Import settings**. Each part goes back to the path in the file, replacing what's there; everything is first backed up to `llmware_data/backups/`. Then **Index all corpora now** indexes every visible corpus with its restored embedding model (or do it later with **⚡ Rescan all corpora**), reload the RAG workspace tab, and turn the connector's and exporter's schedules on again in those apps (an import doesn't install schedules).

### Agents and the outbox

Choose an Agent as the task and **Search**. Its collection prompt (step 1) writes the first answer: a numbered list of what it found, and questions. Then, in the **Discussion** panel:

1. **Refine.** Reply as usual, e.g. *Yes:1,2,3 ignore 4,5,6*, or give due dates and owners. The Agent box under the discussion has a button for each of the Agent's steps (*2 · Select tasks*, *3 · Set due dates*…) that sends that step's prompt for you. Its guidance (how to reply, and the reference snapshot's status) is folded under **How it works**. **Edit prompt workflow** opens this Agent in the prompt library in a new tab, so you can improve its steps as you go; changes apply on your next click. Everything in the discussion can be copied: each message has **Copy**, a step or **Needs work** message also has **Copy prompt sent** (the full text the model received), and **Copy task prompt** under the discussion caption (and beside **Copy answer**) copies the task the model was given, ready to reuse in a flow or a test. **Copy all output**, at the top of the answer (beside **Answer**) and again under it, copies the whole exchange as Markdown: the question, the answer and every follow-up in the discussion.

   **Big screen.** A long reply is easier to read in the main output area than in the discussion panel. **⤢ Last reply on big screen**, beside **Expand** and **Pop out**, shows the latest reply there and keeps following new replies; **⤢ Big screen** under any reply (in the panel or the pop-out) shows that one and stays on it. The question that led to it and the first answer fold beneath, and **↩ Back to the answer** returns the answer to the main area.
2. **Narrow the sources.** Untick a note under **Sources** to leave it out of the following replies and of what the Agent stages. The discussion caption says how many notes are in use.
3. **Execute.** Click **Execute 🤖 Agent:Name with context** at the bottom of the Agent box (blue outline; hover it to see the Action Command). To make a final change without another round in the discussion, type it in the **One last note** box just above the button first, e.g. *exclude points 2-5*. The box is always open and grows as you type; it's sent with the request as a last instruction that overrides the discussion where they differ. The LLM turns what you agreed into exact records, and they're sent straight away as one package (checked against `schemas/handoff.v1.schema.json`), with no second confirmation. Each record becomes its own task, note or email in the other system.
4. **Review while it's held.** A sent package is held for review before any connector can see it: it waits in the outbox's `held/` folder (connectors only read the top level) for the **review time**, 10 minutes unless you change it, and a connector sync in that time doesn't pick it up. The review box under **Execute** shows a one-line summary, when it will be released, and each record. A long body or description is cut short; open **Everything that will be sent** to see every field exactly as it goes into the package. While it's held:
   - **Review time**, at the top of the box: hold this package for 5 to 300 minutes, counted from when it was sent. **Default** opens the Configure page, where the usual review time is set (5 to 60 minutes, under **Outbox**).
   - **✕** drops a record, and choosing a record or answering a check changes it; the held package is rewritten to match.
   - **Send now** releases it into the outbox at once.
   - **Recall** takes it back out: the records stay in the box, marked *Staged — not sent*, to change and **Send** again (held again).
   - **Needs work** recalls it and asks what's wrong; your explanation goes into the discussion (with the rejected list, so the LLM knows what to fix), and you carry on refining and **Execute** again.

   Once released, the package is listed under the box with what happened to it, and **View in outbox** opens the Actions page in a new tab, filtered to that package. Held packages are released by the app while it's running (checked every 20 seconds); if it's stopped, they're released when it starts again.

   **The boxes follow you.** The **Ask a follow-up** box and the Agent box (with **One last note**, **Execute** and the review) stay pinned at the top of the window while you scroll down a long answer, so you can reply or execute without scrolling back. When they're taller than the window, they scroll within themselves.

#### Several Agents at once

Attach more than one Agent, e.g. **Agent:SF_Next_Step** and **Agent:SF_Tasks**, to update Next Steps and create tasks in one action. Each shows as its own token under the question and its own **Task** box; **✕** removes just that one.

- **First answer:** each Agent's collection prompt is a part of one request (*Part 1: SF_Next_Step*, *Part 2: SF_Tasks*), answered under its own heading, with items numbered across all parts so *Yes:1,3 ignore 2* still works.
- **Agent box:** titled *Agent:SF_Next_Step + Agent:SF_Tasks*, with an **Edit** link and a row of step buttons for each Agent; the button reads **Execute 🤖 Agent:SF_Next_Step + Agent:SF_Tasks with context**.
- **Stage:** the model is asked once per Agent (each told it covers only its part), and everything lands in one staging box.
- **Execute** sends one package. When the records differ in type it's a *mixed* `llm-notes-handoff/v2` package, and its prompt reads *SF_Next_Step + SF_Tasks*. The connector must have every type in it switched on (`item_types`), or it skips the whole package.

Unknown values stay empty; nothing is guessed. **Related to** is a name (e.g. *Acme pilot*), not a link; the receiving system finds the matching record. You can execute more than once in the same discussion; each **Execute** is a new package (one still held stays held, and is listed with the packages sent).

**Reference snapshot (real record names).** If the task executor provides one, `outbox/reference/opportunities.json` (format `llm-notes-reference/v1`, see `schemas/reference.v1.schema.json`) lists current records from the other system, e.g. the Salesforce connector's open opportunities with recent activity: name, account, stage, close date, last updated, last activity, description and a link to the record (snapshots from older connectors have no close date; refresh once). This app only reads it; it never contacts the other system. With a snapshot:

- the Agent is given the real names of the records **your notes mention**, and told to use one exactly in **Related to** (or leave it empty rather than invent one), and not to create items for a record just because it's listed. A record counts as mentioned when its name or account appears in a note's text, file name or folder, ignoring case, spaces, punctuation and Ltd/Inc…, so `acme.eu` or `Ac-Me` match the account Acme Ltd. Notes that mention none get no list; choose the record with the picker below;
- in the staging box each item shows **✓** with the record's account, stage and last update when it matches, plus an **Open in Salesforce ↗** link to the record when the snapshot gives one (refresh the snapshot once for older ones). Wherever an opportunity from the snapshot is named (the answer, the discussion, staged items and the Actions page), its name links to it too, or **⚠** when it doesn't (the connector would hold it), and a picker to choose the right record (notes must be linked; tasks can also be standalone);
- **best guess search** links prefer a record whose name or account appears in your question;
- **Duplicate tasks.** The connector also copies **your own open tasks** (`outbox/reference/tasks.json`). Agents that create tasks are shown them (those on the opportunities your notes mention first) and told not to propose one again. In the staging box, a task that looks like one of them shows **⚠ Looks like an open task of yours** with the task, its due date, status, record and an **Open Task ↗** link. Remove it with ✕, or tick **Create anyway**; otherwise the connector holds it rather than create a second one. "Looks like" compares the key words of the subjects (ignoring case, punctuation, the title marker, filler such as *the* or *to*, one- and two-letter words and plural *s*): all of them, or at least two making up three-quarters of the shorter subject, so *Send revised pricing* matches *Send Jo the revised pricing*. Tasks on different records never count. The connector makes the same check against Salesforce right before creating, so nothing slips through between refreshes; a held task shows on the Actions page with **Create anyway and retry**.
- **☁️ Salesforce context** lets any question, not just an Agent, use Salesforce facts, e.g. *give me the link to the Acme opportunity* or *when does it close?*. For each opportunity **the notes mention** (up to 20), the model gets its account, stage, close date, current Next Step, last activity, Salesforce ID and link, plus **your open tasks** on it with their links. It's told to copy links exactly from that list, on their own line, and never invent one. Switch it on in three places:
  - **Ask box:** the **☁️ Salesforce** chip at the start of the **Reference** row. It stays on (and is remembered between sessions) until you click it again.
  - **Discussion:** **📎 Add context: ☁️ Salesforce** above **Ask a follow-up**. It applies from the next follow-up, partway through a chat, and to what an Agent stages. The line beside it says how many opportunities the notes in use mention and when the snapshot was taken. The records are worked out again for each request, so unticking a note drops its records too.
  - **Prompts:** **☁️ Send Salesforce records** on a prompt's card (or in **Add a prompt**) sends them whenever that prompt is used. The Ask chip then shows as on, named after the prompt.
- **Copied text keeps the links.** **Copy**, **Copy answer**, **Copy reply** and **Copy all output** include the opportunity links the app adds on screen, so pasted text links to Salesforce even when the model didn't write the link.
- the Agent box says how many records the snapshot holds and when it was refreshed. Refresh it from the task executor (the Salesforce Connector app refreshes it on every scheduled run and with **Refresh now**).

The snapshot holds customer data, stays in the gitignored outbox folder and is only sent to your local Ollama model. The starter Agent **SF_Opp_Notes** (in the prompt library) summarises what your notes say about each opportunity and stages one note per chosen opportunity, linked to it in Salesforce.

**Next step + chase task.** An Agent with this output updates an opportunity's **Next Step** and creates a task to chase it. For each opportunity it stages two items in one package:

- **Next Step**, signed with today's date and your **Short name** (About me), plus the task title marker, e.g. *2026-10-25-JDOE: waiting on client to respond, will chase on 29 October [AI]*. The staging box shows the new text and the current Next Step from the snapshot (*Now: …*), and warns above 255 characters (the most a Next Step holds) or when no short name is set.
- **A chase task** on the same opportunity, with the due date from your notes or discussion, and the marker.

**Before you send a Next Step,** the staging box shows it as a diff: **Now** is the Next Step in the snapshot (or *not known* when the opportunity isn't in it) and **New** is what it will say, with removed words struck through on red and added words on green. It also checks it against the snapshot:

- **Already says this.** The model is shown each opportunity's current Next Step and marks an update that says the same thing; a near word-for-word match is caught too. The item shows **⚠ The current Next Step already says this** with the current text, so you can remove it (and its chase task) with ✕, or keep it to restate it.
- **Changed since you staged it.** When the snapshot is refreshed after you staged and the Next Step is now different, the item shows **⚠ Changed since you staged this** with the new text. **Replace it anyway** sends yours as a deliberate overwrite of that new text; otherwise the connector holds it (below).

The connector writes the Next Step only if the opportunity still has the one you saw; if someone changed it since, it holds the update as *changed since staged* (refresh the snapshot and stage again), or use **Overwrite anyway and retry** on the Actions page to replace whatever is there now). These packages use format `llm-notes-handoff/v2` (`schemas/handoff.v2.schema.json`): v1 plus update items and mixed packages. On the Actions page a waiting Next Step can be edited like any item. The starter Agent **SF_Next_Step** asks which opportunities to update and when to chase.

The app never contacts another system. A separate connector (for example [rag-action-interface](https://github.com/yrros/rag-action-interface) for Salesforce) reads the outbox, moves each package to `outbox/processed/` or `outbox/failed/`, and leaves a `<package name>.receipt.json` beside it. The Agent box shows each sent package's result once its receipt appears: what was created, with links to each record and the record it's on when the connector gives them (**Open Task ↗**, **Open the record it's on ↗**), and why anything wasn't. The Actions page shows the same links beside each record.

**Outbox status line.** The top of every page shows, e.g., *Outbox: 2 waiting (7 items) · 1 held for review · 1 needs attention · Last processed 25 Oct 10:42 · Last error 24 Oct 18:03 · Open Actions*. "Waiting" counts packages not yet picked up and the records in them; "needs attention" counts packages in `failed/`, including items held for you to choose a record; the dates come from the connector's receipts.

**Actions page.** **Open Actions** at the end of the status line opens `?page=actions` in a new tab: everything in the outbox, newest first, in five tabs — **Needs attention** (packages in `failed/`, opened up), **Held for review** (with **Send now**, **Withdraw** and when each is released), **Waiting**, **Processed** and **Withdrawn** — plus the [Actions log](#actions-log). Each package shows the Agent that sent it, its library, question, model and source notes, the connector's overall result, and every record with its outcome: the record ID it created, or why it didn't (with possible matches when a name was ambiguous). Filter by any text, e.g. a subject, owner or client name; **Refresh** re-reads the folder. Under the title, **Task executor** links to the service that processes the outbox (set in **Task executor settings** on the **Configure** page), with 🟢 running / 🔴 not running.

#### Actions log

The **Actions log** tab on the Actions page tracks which notes each Agent has actioned, so you can tell what's already in Salesforce:

- **Sent (⏳).** **Execute** (or **Send**) logs every note in use against each Agent that staged items from it. **Recall** removes those entries again.
- **Done (✓).** Once the connector's receipt says at least one of that Agent's items was created, updated or already existed, the entry is done, with a link to the record.
- **Failed (⚠) or withdrawn (↩).** The package failed, or was withdrawn or deleted on the Actions page. A retried package that then goes through becomes done.
- **Done by hand (✓, mark).** Type an Agent's **Actioned mark** (set on its card in the Prompt library, `[SF_Tasks]` by default) anywhere in a note in Apple Notes. Once the exporter has exported the note and a rescan (auto or manual) has indexed it, the note is logged as done by that Agent. Marks escaped by a Markdown export, such as `\[SF\_Tasks\]`, count too. The entry stays if you later remove the mark.

Nothing is written back to Apple Notes or the exported files. In search results and the in-scope list, each note's title shows its Agents (` · ✓ SF_Tasks`), and opening it shows how and when, with a link to the record. The **Actioned** control under **Max results** in the Filter box shows **All notes**, **Not yet actioned** or **Only actioned**, by any Agent or one. It applies to every search route, and the filter line shows it. Set to anything but **All notes**, the control turns strongly red, so you don't miss that notes are being left out; **Clear all** sets it back to **All notes**. Sent and done count as actioned; failed and withdrawn don't.

On the tab, filter by Agent, status or note name, select rows to delete them (a note still holding its mark is logged again on the next scan), or **Scan notes for marks now** to look without waiting for a rescan. Marks are looked for in every visible corpus (hidden ones are skipped) when a corpus has been rescanned since its last scan or when a mark changes. The log is kept in `llmware_data/actions_log.json`. Back it up and restore it under **Advanced — Actions log** in the sidebar.

**Getting back.** On every screen other than the main one (the Prompt library, the Actions page and the Salesforce Connector app) the 🧠 before the title takes you back to the main screen in the same tab.

| Button | On | What it does |
|--------|----|--------------|
| **Withdraw** | Waiting, Needs attention | Asks to confirm, then moves the package (and its receipt) to `outbox/withdrawn/`, where connectors don't look. Nothing already created in the other system is undone. |
| **Retry** | Needs attention | Removes the receipt and moves the package back to the outbox for the connector's next run, e.g. after a temporary error. Records already created aren't created twice. Disabled when the file itself is invalid. |
| **Restore** | Withdrawn | Moves the package back to the outbox for a connector to process. |
| **Delete** | Withdrawn | Deletes the package and its receipt for good, after you confirm. **Delete all** at the top of the tab does the same for every withdrawn package listed (only those matching the filter, when there is one). Only files in `outbox/withdrawn/` can be deleted. |
| **Edit** | Waiting, Needs attention, Withdrawn | Opens the package's items for editing: subject or title, description or body, due date, priority, owner, recipients and **Related to**. **Save changes** checks the package against the schema and rewrites the file atomically, keeping its id (so nothing is ever created twice). Use it to add a marker such as `[TEST]` to find records easily, or to fix a name the connector couldn't match, then **Retry**. On a partly done package, records already created aren't shown, since editing them wouldn't change the other system. If a connector picks the package up while you edit, nothing is saved. |

#### Fixing an item with context

Every item that can still be edited (on **Needs attention**, **Waiting** and **Withdrawn**, and not already created) has a **Fix with context** switch. It opens an inline panel:

1. **Extra context.** Type what the model needs to know, e.g. *Joseph Oliver is known as J-O*.
2. **Re-assess** is a dry run, in two steps:
   - **Names.** The model reads the Agent Semantic Fixes, **About me**, the library's corpus prompt and your context, and lists each nickname with the full name it stands for. It's also given the names in the item and asked which are short for a full name there. A name it leaves unmatched that starts the first name of exactly one full name in those facts (e.g. *Sam* and *Samantha Example* in About me) is matched by the app; if two people could fit, it isn't. The app then swaps any person field (owner, recipients) that is exactly a listed nickname for the full name. The swap doesn't depend on the model judging the item, and Salesforce matches people by full name only. **The same swap runs when an Agent stages its items**, so a package goes out with full names in the first place rather than failing on a nickname from the notes.
   - **Everything else.** The model gets the item (names already swapped), the connector's message about it, your context, every fix and the package's source notes, re-read from disk (up to 8,000 characters shared between them). It returns the item corrected, with a one-line reason. It can't undo a name swap.

   Each change is listed as old → new with a tickbox, marked *full name from the fixes* when it came from the first step. Untick any you don't want. Nothing is saved yet. Re-assess uses the **Model** picked for the duplicate check at the top of the page.
3. **Apply and retry** saves the ticked changes in the package (checked against the schema, same package id) and sends it back to the outbox for the connector's next run. A withdrawn package goes back too, and records already created aren't created twice. **Discard** drops the proposal.
4. If you typed context, a pop-up then offers to add it to **Agent Semantic Fixes**, editable first, so the next Re-assess knows it without retyping.

In testing on a held task, `llama3.2:3b`, `qwen3:30b` and `qwen3.8:27b` all swapped a nickname owner for the full name from a saved fix. Asked to judge the item alone, `qwen3:30b` left the nickname in place, which is why names are swapped by the app. The smaller models sometimes also reword another field, so check the tickboxes.

If the connector reports a problem only on screen (e.g. `sf apply` printing HOLD) and leaves no receipt, Re-assess doesn't see the message. Paste it into **Extra context**.

When a record needs choosing (an ambiguous or missing name), either choose it in the connector, which retries the package by itself, or correct **Related to** with **Edit** and press **Retry**. A package you withdraw also shows as withdrawn in the Agent box of the discussion that sent it.

**Checking for duplicates.** Each item on the **Waiting** and **Withdrawn** tabs has a **Check in Salesforce** line: a link to the record it's on, and links to your open tasks on that record (from the connector's snapshot), so you can see what's already there. **🔍 Check for duplicates**, above the tabs, has your local model compare each Waiting and Withdrawn item with:

- the other items in the outbox, in any tab (so the same task sent twice, or one already processed, shows up);
- for tasks, your open tasks in Salesforce.

Only likely pairs go to the model: items of the same kind on the same record, or sharing a key word when one has no record (up to 12 per item, same record first). The model gives a same/different verdict on each, judging by meaning, so *Send revised pricing* and *Email the updated price list* match but *Book kickoff workshop* doesn't. Likely duplicates are marked **⚠ Looks like …** under the item with what they match, where it is (or an **Open Task ↗** link) and the model's reason. The tab and the package heading show a ⚠ count too. Nothing is changed: withdraw or keep each one yourself. The results last for the session and disappear from an item once you edit it.

Pick the **Model** beside the button. It starts on the largest model you have installed, because small ones judge poorly: in testing, `llama3.2:3b` missed rewordings and flagged different actions, while `qwen3:30b` got them right. Reasoning models are asked to answer without thinking, which is as accurate here and far faster (about 20 seconds for a handful of items with `qwen3:30b`).

**Default location:** `outbox/` in the project folder, i.e. the folder you run Streamlit from, which is the project folder when you start with `start_app.sh` or the Desktop launcher. Change it under **Outbox** on the Prompt library page (saved as `outbox_path` in `app_settings.json`); the connector must read the same folder. The **Review time before sending** slider beside it sets how long a sent package is held in `held/` first (see the Agent steps above). The outbox is gitignored because sent drafts contain your notes, and `reset_rag.sh` leaves it alone.

Each file is named `YYYYMMDD-HHMMSS-<note|task|email>-<first 8 characters of the package id>.json` and written atomically (a hidden `.<name>.tmp` file is renamed into place).

### Advanced settings

| Setting | Scope | Notes |
|---------|-------|-------|
| Also use GPU for embeddings | Global | Uses CUDA when available, or the Apple Silicon GPU (MPS) on Mac. Falls back to CPU if a GPU op fails. |
| Auto-rescan every 3 min (text only) | Global | While the Streamlit server runs (with or without a browser tab), rescans the selected corpus for new, edited and deleted notes and updates the text index, so filters and literal search find them. Embeddings are not built — semantic search sees the changes after **⚡ Rescan**. Starts on the first page load after the server starts. |
| …in every corpus | Global | Auto-rescan every visible corpus instead of just the selected one. Hidden corpora are skipped. |
| Manage corpus visibility | Global | In the sidebar's **Configure** dropdown (either corpus mode). Hides corpora from the library list **and** from every all-corpora scan and rebuild, including auto-rescan. A hidden corpus keeps its index as it was; select it to rescan it on its own. Only applies while **Enable corpus hiding** is ticked. |
| Embedding engine | Per corpus | Changing model requires **Re-index** after acknowledging the warning. |
| Relevance cutoff (distance) | Per corpus | Lower = stricter matches. Saved per corpus; no re-embed needed. |
| Back up / restore actions log | Global | Under **Advanced — Actions log**. **Back up actions log** downloads the log as JSON (`llm-notes-actions-log/v1`). **Restore** reads a backup: **Merge** adds entries you don't have, **Replace** swaps the log for it. Either way the log as it was is saved to `llmware_data/backups/` first, and marks are scanned again. The log isn't in the Configure page's settings export, because it names notes. |
| Debug details | Per session | Paths, file counts, index health |

### Rescan vs Re-index

| Action | Scope | When to use |
|--------|-------|-------------|
| **⚡ Rescan (incremental)** | This corpus | New, edited or deleted notes; same embedding model. Embeds only the new blocks. |
| **🔨 Re-index (full rebuild)** | This corpus | First index, corrupted index, or embedding model change. Asks first: it warns that searching this library won't work until it's finished, says how many notes it will rebuild and how long the last full rebuild took here, points to **Rescan (incremental)** for everyday changes, and needs a tick before **Rebuild now** |
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
├── actions_log.json      # Which notes each Agent has actioned (Actions log)
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
- `outbox_hold_minutes` — the default review time (5–60, default 10): how long a sent package waits in `held/` before it's released into the outbox for a connector
- `actioned_filter` — the Filter box's **Actioned** control: `mode` (`all`, `hide` or `only`) and `agent` (a prompt id, or empty for any Agent)
- `note_dates` — how date filters find a note: `text` (a date written in it), `created` (its file's creation date) or `either`
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
| Wrong or missing corpus root | Path not set or folder moved | Set **Notes location** under **Configure** in the sidebar (it opens by itself); use **Browse…** |
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
| `start_app.sh` | Start/warm Ollama, then launch Streamlit on port 8510 (or open it if it's already running) |
| `llm_notes_service.sh` | Run the app in the background when opened (install, open, stop, restart, status, logs) and build the Desktop icon |
| `stop_app.sh` | Stop the app and free its memory, however it was started |
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
