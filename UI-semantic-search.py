import math
import re
import json
import gc
import hashlib
import shutil
import sqlite3
import sys
import time
import threading
import datetime
import difflib
import html
import http.client
import itertools
import queue
import socket
import uuid
from contextlib import contextmanager
from functools import lru_cache
from urllib.parse import quote_plus, urlparse
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from llmware.library import Library, LibraryCatalog
from llmware.retrieval import Query
from llmware.resources import Status
from llmware.models import ModelCatalog, HFEmbeddingModel
from llmware.configs import LLMWareConfig, LanceDBConfig, SQLiteConfig
from llmware.embeddings import _EmbeddingUtils

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "mini-lm-sbert"
DEFAULT_DISTANCE_THRESHOLD = 0.55
VECTOR_DB = "lancedb"
# Local data directory — everything lives under the folder you run Streamlit from.
PROJECT_DIR = Path.cwd()
APP_DATA_DIR = PROJECT_DIR / "llmware_data"
LEGACY_APP_DATA_DIR = Path.home() / "llmware_data"
LANCEDB_PATH = APP_DATA_DIR / "lancedb"
LEGACY_LANCEDB_PATH = Path("/tmp/lancedb")
EMBEDDING_POLL_SECONDS = 5
AUTO_RESCAN_SECONDS = 180
INGEST_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".json"}

RECOMMENDED_EMBEDDING_MODELS = [
    "mini-lm-sbert",
    "bge-small-en-v1.5",
    "bge-base-en-v1.5",
    "bge-large-en-v1.5",
    "nomic-text-v1",
    "gte-base",
    "gte-large",
    "mpnet-base",
    "jina-base-en-v2",
    "jina-small-en-v2",
    "ember-v1",
]
EMBEDDING_MODEL_SUMMARIES = {
    "mini-lm-sbert": "Fast default. Lightest and quickest to embed.",
    "bge-small-en-v1.5": "Better quality than Mini LM, still relatively fast.",
    "bge-base-en-v1.5": "Strong general-purpose search. Good speed vs. quality balance.",
    "bge-large-en-v1.5": "Best BGE retrieval quality. Slower to embed.",
    "nomic-text-v1": "High-quality open embeddings with strong semantic matching.",
    "gte-base": "Reliable general text retrieval with a good balance of speed and quality.",
    "gte-large": "High-quality GTE model for nuanced semantic search. Slower to embed.",
    "gte-small": "Compact and fast. A small step up from Mini LM.",
    "mpnet-base": "Classic sentence embeddings. Solid all-rounder.",
    "jina-base-en-v2": "Strong modern embeddings with good semantic understanding.",
    "jina-small-en-v2": "Lighter Jina model. Faster, with decent search quality.",
    "ember-v1": "High-quality embeddings tuned for retrieval tasks.",
    "uae-large-v1": "Large universal embeddings. High quality, slower indexing.",
    "industry-bert-sec": "Domain-tuned for SEC filings and financial documents.",
    "industry-bert-contracts": "Domain-tuned for contracts and legal language.",
    "industry-bert-insurance": "Domain-tuned for insurance documents and terminology.",
    "industry-bert-loans": "Domain-tuned for loan and lending documents.",
    "industry-bert-asset-management": "Domain-tuned for asset management and investment text.",
}
EXCLUDED_MODEL_KEYWORDS = (
    "reranker",
    "protectai",
    "toxic",
    "bias",
    "language-detector",
    "prompt-injection",
)
LOCAL_EMBEDDING_FAMILIES = {"HFEmbeddingModel", "LLMWareSemanticModel"}
# Optional pip packages required by specific embedding models (not installed by default).
EMBEDDING_MODEL_EXTRA_DEPS: dict[str, tuple[str, ...]] = {
    "nomic-text-v1": ("einops",),
}
# Extra system install steps shown after pip install for optional ingest/parser dependencies.
OPTIONAL_DEPENDENCY_EXTRA_LINES: dict[str, list[str]] = {
    "pytesseract": ["brew install tesseract  # system OCR engine (Mac); needed after pip install"],
}
# Map LLMWare dependency names to pip package names.
PIP_PACKAGE_ALIASES: dict[str, str] = {
    "sentence_transformer": "sentence-transformers",
    "sentence_transformers": "sentence-transformers",
}

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
SUGGESTED_OLLAMA_MODELS = [
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3.1:8b",
    "mistral",
    "phi3:mini",
    "gemma2:2b",
]
LLM_CONTEXT_MAX_CHARS = 12000
# Follow-up conversations resend the notes plus history, so give Ollama a larger window.
FOLLOW_UP_NUM_CTX = 8192
# Ollama's context window must hold the prompt and the whole answer. These sizes are used in steps, so
# the model isn't reloaded for every small change; bigger windows use more memory.
OLLAMA_CONTEXT_SIZES = (8192, 16384, 32768, 65536)
LLM_PROGRESS_SECONDS = 3  # how often the tokens-per-second count refreshes while an answer streams
# How often the page is touched while waiting on Ollama: Streamlit only acts on a click (Cancel) when the
# running script updates the page.
LLM_TICK_SECONDS = 0.5
LLM_CANCELLED_KEY = "llm_cancelled"
# Max tokens: up to roughly a short novel. English runs at about 0.75 words per token.
WORDS_PER_TOKEN = 0.75
LLM_MAX_TOKENS_LIMIT = 50000
BOOK_LENGTHS = (  # (title, approximate word count): markers on the Max tokens slider
    ("The Very Hungry Caterpillar", 225),
    ("Where the Wild Things Are", 340),
    ("The Tale of Peter Rabbit", 950),
    ("Dr. Seuss, The Cat in the Hat", 1600),
    ("The Little Prince", 17000),
    ("Winnie-the-Pooh", 23000),
    ("The Old Man and the Sea", 27000),
    ("Animal Farm", 30000),
    ("The Lion, the Witch and the Wardrobe", 37000),
)
BOOKS_BEYOND_LIMIT = (("The Hobbit", 95000), ("War and Peace", 587000))
TOKEN_STEPS = (128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 40960, LLM_MAX_TOKENS_LIMIT)
DISCUSSION_HEIGHT = 520
DISCUSSION_EXPANDED_HEIGHT = 1100
FOLLOW_UP_BOX_HEIGHT = 90
FOLLOW_UP_BOX_EXPANDED_HEIGHT = 320
LAST_SEARCH_KEY = "last_search"
CORPUS_SETTINGS_PATH = APP_DATA_DIR / "corpus_settings.json"
JOB_STATUS_PATH = APP_DATA_DIR / "job_status.json"
APP_SETTINGS_PATH = APP_DATA_DIR / "app_settings.json"
PROMPT_LIBRARY_PATH = APP_DATA_DIR / "prompt_library.json"
TEXT_RESCAN_STATUS_PATH = APP_DATA_DIR / "text_rescan_status.json"
ACTIONS_LOG_PATH = APP_DATA_DIR / "actions_log.json"  # which notes each Agent has actioned
PROMPT_STACK_SIZE = 5

DEFAULT_APP_SETTINGS = {
    "use_gpu_for_embeddings": False,
    "hide_corpora_enabled": False,
    "corpus_mode": "multi",
    "last_selected_library": None,
    "auto_rescan_enabled": False,
    "auto_rescan_all_corpora": False,
}

CORPUS_MODE_MULTI = "multi"
CORPUS_MODE_SINGLE = "single"
SINGLE_CORPUS_DISPLAY_NAME = "All notes"
SINGLE_CORPUS_LIBRARY_NAME = "Single_Corpus"
LOCAL_NOTES_EXPORT_URL = "https://github.com/yrros/Local-Apple-Notes-Export"
FALCON_NOTES_EXPORTER_URL = "https://falcon.star-lord.me/exporter"
SETUP_TRIAL = "trial"
SETUP_WHOLE = "whole"
SETUP_PER_FOLDER = "per_folder"

DEFAULT_CORPUS_CONFIG = {
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "distance_threshold": DEFAULT_DISTANCE_THRESHOLD,
    "vector_db": VECTOR_DB,
    "ollama_model": DEFAULT_OLLAMA_MODEL,
    "llm_max_tokens": 512,
    "llm_temperature": 0.3,
    "hidden": False,
    "corpus_prompt": "",
    "corpus_prompt_enabled": False,
}

# Hardware acceleration
# - Ollama (Answer with LLM) uses its own Metal/GPU stack — independent of LLMWare config.
# - LLMWare HF embedding models only check CUDA by default; patch below routes to CUDA or MPS when enabled.
LLMWareConfig().set_config("hardware_accelerator", "mac_metal")
LLMWareConfig().set_home(str(PROJECT_DIR))
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
LANCEDB_PATH.mkdir(parents=True, exist_ok=True)
# SQLiteConfig captures get_library_path() at import time — refresh after set_home().
SQLiteConfig.set_config("sqlite_db_folder_path", str(APP_DATA_DIR / "accounts"))
LanceDBConfig.set_config("uri", str(LANCEDB_PATH))


def apply_llmware_python314_sqlite_fix() -> None:
    """LLMWare 0.4.x uses $1…$N SQLite placeholders with tuple bindings; Python 3.14 requires ?."""
    if sys.version_info < (3, 14):
        return

    from llmware import resources as llmware_resources

    writer_cls = llmware_resources.SQLiteWriter
    if getattr(writer_cls, "_python314_sqlite_patched", False):
        return

    def write_new_parsing_record(self, rec):
        sql_string = f"INSERT INTO {self.library_name}"
        sql_string += (
            " (block_ID, doc_ID, content_type, file_type, master_index, master_index2, "
            "coords_x, coords_y, coords_cx, coords_cy, author_or_speaker, added_to_collection, "
            "file_source, table_block, modified_date, created_date, creator_tool, external_files, "
            "text_block, header_text, text_search, user_tags, special_field1, special_field2, "
            "special_field3, graph_status, dialog, embedding_flags) "
            "VALUES (" + ", ".join(["?"] * 28) + ");"
        )
        insert_arr = (
            rec["block_ID"],
            rec["doc_ID"],
            rec["content_type"],
            rec["file_type"],
            rec["master_index"],
            rec["master_index2"],
            rec["coords_x"],
            rec["coords_y"],
            rec["coords_cx"],
            rec["coords_cy"],
            rec["author_or_speaker"],
            rec["added_to_collection"],
            rec["file_source"],
            rec["table"],
            rec["modified_date"],
            rec["created_date"],
            rec["creator_tool"],
            rec["external_files"],
            rec["text"],
            rec["header_text"],
            rec["text_search"],
            rec["user_tags"],
            rec["special_field1"],
            rec["special_field2"],
            rec["special_field3"],
            rec["graph_status"],
            rec["dialog"],
            "",
        )
        self.conn.cursor().execute(sql_string, insert_arr)
        self.conn.commit()
        self.conn.close()
        return True

    writer_cls.write_new_parsing_record = write_new_parsing_record
    writer_cls._python314_sqlite_patched = True


apply_llmware_python314_sqlite_fix()


def apply_llmware_fts_query_fix() -> None:
    """LLMWare FTS prep keeps empty tokens from trailing spaces, producing invalid `van OR` queries."""
    from llmware import resources as llmware_resources

    retrieval_cls = llmware_resources.SQLiteRetrieval
    if getattr(retrieval_cls, "_fts_query_patched", False):
        return

    def _prep_query(self, query):
        if re.search(r"\s+AND\s+", query, flags=re.IGNORECASE):
            return query
        if re.search(r"\s+OR\s+", query, flags=re.IGNORECASE):
            return query
        sqlite_strings = {"AND": " AND ", "OR": " OR "}
        exact_match = query.startswith('"') and query.endswith('"')
        q_clean = re.sub(r"[^\w\s]", "", query)
        q_toks = [tok for tok in q_clean.split() if tok]
        if not q_toks:
            return ""
        joiner = sqlite_strings["AND"] if exact_match else sqlite_strings["OR"]
        return joiner.join(q_toks)

    def basic_query(self, query):
        query_str = self._prep_query(query)
        if not query_str.strip():
            return []
        sql_query = (
            f"SELECT rank, rowid, * FROM {self.library_name} "
            f"WHERE text_search MATCH '{query_str}' ORDER BY rank"
        )
        results = self.conn.cursor().execute(sql_query)
        output = self.unpack_search_result(results)
        self.conn.close()
        return output

    retrieval_cls._prep_query = _prep_query
    retrieval_cls.basic_query = basic_query
    retrieval_cls._fts_query_patched = True


apply_llmware_fts_query_fix()


def apply_llmware_ocr_error_fix() -> None:
    """Skip note images that crash Tesseract instead of aborting the whole corpus scan."""
    from llmware.parsers import ImageParser

    if getattr(ImageParser, "_ocr_error_patched", False):
        return

    _original_process_ocr = ImageParser.process_ocr

    def process_ocr_resilient(self, dir_fp, fn, preserve_spacing=False):
        try:
            return _original_process_ocr(self, dir_fp, fn, preserve_spacing=preserve_spacing)
        except Exception as exc:
            exc_name = type(exc).__name__
            if exc_name in {"TesseractError", "TesseractNotFoundError"}:
                return []
            raise

    ImageParser.process_ocr = process_ocr_resilient
    ImageParser._ocr_error_patched = True


apply_llmware_ocr_error_fix()


def raise_open_file_limit() -> None:
    """Raise the soft open-file limit so large full-corpus ingests do not hit EMFILE."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 65_536)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass


raise_open_file_limit()

_EMBEDDING_USE_GPU = False
_EMBEDDING_PATCH_INSTALLED = False
_LANCEDB_CONN = None


def get_lancedb_connection():
    """Reuse one LanceDB connection — avoids leaking file handles on repeated polls."""
    global _LANCEDB_CONN
    if _LANCEDB_CONN is None:
        import lancedb

        _LANCEDB_CONN = lancedb.connect(str(LANCEDB_PATH))
    return _LANCEDB_CONN


def migrate_legacy_app_data_if_needed() -> list[str]:
    """One-time copy from ~/llmware_data when the project-local store is still empty."""
    if not LEGACY_APP_DATA_DIR.exists():
        return []

    local_ready = (
        CORPUS_SETTINGS_PATH.exists()
        or any(LANCEDB_PATH.glob("*.lance"))
        or (APP_DATA_DIR / "accounts").exists()
    )
    if local_ready:
        return []

    migrated: list[str] = []
    for rel_path in (
        "corpus_settings.json",
        "job_status.json",
        "app_settings.json",
        "lancedb",
        "accounts",
        "model_repo",
    ):
        src = LEGACY_APP_DATA_DIR / rel_path
        dest = APP_DATA_DIR / rel_path
        if not src.exists() or dest.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        migrated.append(rel_path)
    return migrated


_migrated_legacy_app_data = migrate_legacy_app_data_if_needed()


def load_app_settings() -> dict:
    settings = DEFAULT_APP_SETTINGS.copy()
    if not APP_SETTINGS_PATH.exists():
        return settings
    try:
        settings.update(json.loads(APP_SETTINGS_PATH.read_text()))
    except Exception:
        pass
    return settings


def save_app_settings(updates: dict) -> dict:
    settings = load_app_settings()
    settings.update(updates)
    APP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    return settings


def load_corpus_root_path() -> Path | None:
    """Load the saved corpus root from app settings."""
    raw = load_app_settings().get("corpus_root_path")
    if not raw or not str(raw).strip():
        return None
    return Path(str(raw)).expanduser()


def get_corpus_root_path() -> Path | None:
    """Return the active corpus root (widget state, then saved settings)."""
    raw = st.session_state.get("corpus_root_path_input")
    if raw is not None:
        text = str(raw).strip()
        if text:
            return Path(text).expanduser()
    saved = load_corpus_root_path()
    if saved:
        return saved
    return None


def persist_corpus_root_path() -> None:
    raw = str(st.session_state.get("corpus_root_path_input", "")).strip()
    if raw:
        save_app_settings({"corpus_root_path": raw})


def sync_corpus_root_widget_from_settings() -> None:
    """Keep the folder text field aligned with saved settings when widget state is blank."""
    saved = load_corpus_root_path()
    if not saved:
        return
    saved_text = str(saved)
    if not str(st.session_state.get("corpus_root_path_input", "")).strip():
        st.session_state["corpus_root_path_input"] = saved_text


def ensure_corpus_root_persisted(corpus_root: Path | None) -> None:
    """Write the active corpus root to app settings before long-running index jobs."""
    if corpus_root is None:
        return
    save_app_settings({"corpus_root_path": str(corpus_root)})


def get_corpus_mode() -> str:
    """Return active corpus layout mode: multi (subfolder per library) or single (whole tree)."""
    mode = st.session_state.get("corpus_mode_toggle")
    if mode in (CORPUS_MODE_MULTI, CORPUS_MODE_SINGLE):
        return mode
    saved = load_app_settings().get("corpus_mode", CORPUS_MODE_MULTI)
    return saved if saved in (CORPUS_MODE_MULTI, CORPUS_MODE_SINGLE) else CORPUS_MODE_MULTI


def is_single_corpus_mode() -> bool:
    return get_corpus_mode() == CORPUS_MODE_SINGLE


def persist_corpus_mode() -> None:
    mode = st.session_state.get("corpus_mode_toggle", CORPUS_MODE_MULTI)
    if mode not in (CORPUS_MODE_MULTI, CORPUS_MODE_SINGLE):
        mode = CORPUS_MODE_MULTI
    save_app_settings({"corpus_mode": mode})


def sync_corpus_mode_from_settings() -> None:
    if "corpus_mode_toggle" not in st.session_state:
        st.session_state["corpus_mode_toggle"] = load_app_settings().get("corpus_mode", CORPUS_MODE_MULTI)


def load_last_selected_library() -> str | None:
    saved = load_app_settings().get("last_selected_library")
    return str(saved).strip() if saved else None


def persist_selected_library() -> None:
    selected = st.session_state.get("selected_library")
    if selected:
        save_app_settings({"last_selected_library": selected})


def build_library_select_options(all_libraries: list[str]) -> list[str]:
    """Libraries shown in the selector, keeping the last-used corpus even if hidden."""
    visible = filter_libraries_for_display(all_libraries)
    if not visible:
        visible = list(all_libraries)

    saved = load_last_selected_library()
    if saved and saved in all_libraries and saved not in visible:
        visible = [saved] + [lib for lib in visible if lib != saved]
    return visible


def sync_selected_library_from_settings(visible_libraries: list[str]) -> None:
    if "selected_library" in st.session_state:
        return
    saved = load_last_selected_library()
    if saved and saved in visible_libraries:
        st.session_state["selected_library"] = saved


def resolve_library_registry_name(selected_lib: str) -> str:
    """LLMWare library key (sanitized) for SQLite and LanceDB."""
    if is_single_corpus_mode():
        return SINGLE_CORPUS_LIBRARY_NAME
    return sanitize_library_name(selected_lib)


def resolve_library_content_path(corpus_root: Path, selected_lib: str) -> Path:
    """Folder on disk that is scanned during ingest for the active selection."""
    if is_single_corpus_mode():
        return corpus_root
    return corpus_root / selected_lib


def corpus_has_ingestible_files(corpus_root: Path) -> bool:
    """True when the corpus root contains at least one supported ingest file."""
    for path in corpus_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in INGEST_EXTENSIONS:
            return True
    return False


def browse_for_directory(initial_dir: str | None = None) -> str | None:
    """Open a native folder picker. Uses AppleScript on macOS, tkinter elsewhere."""
    import platform
    import subprocess

    if platform.system() == "Darwin":
        script = 'POSIX path of (choose folder with prompt "Select notes export folder")'
        if initial_dir:
            initial = Path(initial_dir).expanduser()
            if initial.exists():
                escaped = str(initial).replace("\\", "\\\\").replace('"', '\\"')
                script = (
                    'POSIX path of (choose folder with prompt "Select notes export folder" '
                    f'default location (POSIX file "{escaped}"))'
                )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        picked = result.stdout.strip().rstrip("/")
        return picked or None

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
    except Exception:
        pass

    kwargs: dict = {"title": "Select notes export folder", "mustexist": True}
    if initial_dir and Path(initial_dir).expanduser().exists():
        kwargs["initialdir"] = str(Path(initial_dir).expanduser())

    try:
        selected = filedialog.askdirectory(**kwargs)
    finally:
        root.destroy()
    return selected or None


def _pick_embedding_device(use_gpu: bool) -> str:
    """Choose the torch device for HF embedding models."""
    if not use_gpu:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU; embedding falls back to CPU if an MPS op fails mid-scan.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_embedding_device_label(use_gpu: bool) -> str:
    return _pick_embedding_device(use_gpu)


def _run_hf_embedding_forward(model, input_ids, attn_mask, device: str):
    import torch

    run_device = device
    try:
        if run_device != "cpu":
            model.to(run_device)
            input_ids = input_ids.to(run_device)
            attn_mask = attn_mask.to(run_device)
        else:
            model.to("cpu")
            input_ids = input_ids.to("cpu")
            attn_mask = attn_mask.to("cpu")
        with torch.no_grad():
            return model(input_ids, attention_mask=attn_mask), run_device
    except RuntimeError as exc:
        message = str(exc)
        if run_device == "cpu" or (
            "MPS" not in message and "Placeholder storage" not in message
        ):
            raise
        with torch.no_grad():
            model.to("cpu")
            return model(
                input_ids.to("cpu"),
                attention_mask=attn_mask.to("cpu"),
            ), "cpu"


def _install_embedding_acceleration_patch() -> None:
    global _EMBEDDING_PATCH_INSTALLED
    if _EMBEDDING_PATCH_INSTALLED:
        return

    original_init = HFEmbeddingModel.__init__

    def init_with_accel(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.model is None:
            self.use_gpu = False
            self._embedding_device = "cpu"
            return

        device = _pick_embedding_device(_EMBEDDING_USE_GPU)
        self._embedding_device = device
        self.use_gpu = device != "cpu"
        if self.use_gpu:
            import torch

            self.model.to(device)
        else:
            self.model.to("cpu")

    def embedding_with_accel(self, text_sample, api_key=None):
        import numpy as np
        import torch

        self.text_sample = text_sample
        self.preview()

        sequence = self.text_sample if isinstance(self.text_sample, list) else [self.text_sample]
        model_inputs = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            padding=True,
        )

        preferred = getattr(self, "_embedding_device", "cpu")
        model_outputs, used_device = _run_hf_embedding_forward(
            self.model,
            model_inputs.input_ids,
            model_inputs.attention_mask,
            preferred,
        )
        if used_device != preferred:
            self._embedding_device = used_device
            self.use_gpu = used_device != "cpu"

        embedding = model_outputs.last_hidden_state[:, 0]
        embeddings_normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
        embeddings_normalized = np.array(embeddings_normalized.detach().to("cpu"))

        self.register()
        return embeddings_normalized

    HFEmbeddingModel.__init__ = init_with_accel
    HFEmbeddingModel.embedding = embedding_with_accel
    _EMBEDDING_PATCH_INSTALLED = True


def set_embedding_acceleration(use_gpu: bool) -> str:
    """Enable or disable GPU/MPS for LLMWare embedding models."""
    global _EMBEDDING_USE_GPU
    _install_embedding_acceleration_patch()
    _EMBEDDING_USE_GPU = bool(use_gpu)
    return resolve_embedding_device_label(_EMBEDDING_USE_GPU)


def get_use_gpu_for_embeddings() -> bool:
    return bool(
        st.session_state.get(
            "use_gpu_for_embeddings_global",
            load_app_settings().get("use_gpu_for_embeddings", False),
        )
    )


def persist_gpu_for_embeddings_setting() -> None:
    use_gpu = bool(st.session_state.get("use_gpu_for_embeddings_global", False))
    save_app_settings({"use_gpu_for_embeddings": use_gpu})
    set_embedding_acceleration(use_gpu)


set_embedding_acceleration(load_app_settings().get("use_gpu_for_embeddings", False))


SEARCH_ROUTES = {"semantic": "Semantic (meaning)", "literal": "Literal (exact text)"}


def saved_search_route() -> str:
    """The route last chosen; stored as `exact_text_only` so older settings carry over."""
    return "literal" if load_app_settings().get("exact_text_only", False) else "semantic"


MAX_RESULTS_LIMITS = {"semantic": 50, "literal": 1000}  # passages (Semantic) or notes listed (Literal)
ASK_NOTES_LIMIT = 50  # notes Ask (filtered only) answers from, whatever Max results says: the LLM reads few anyway


def max_results_key(literal: bool) -> str:
    """Each route keeps its own Max results, so a big literal listing doesn't carry over to Semantic."""
    return "max_results_literal" if literal else "max_results"


def persist_max_results(key: str = "max_results") -> None:
    save_app_settings({key: int(st.session_state.get(key, 15))})


def persist_search_route() -> None:
    route = st.session_state.get("search_route")
    if route is None:  # clicking the selected option again clears it: keep the route instead
        st.session_state["search_route"] = saved_search_route()
        return
    save_app_settings({"exact_text_only": route == "literal"})


DISTANCE_SLIDER_MIN = 0.30
DISTANCE_SLIDER_MAX = 1.20
DISTANCE_SLIDER_STEP = 0.05


def cutoff_to_include(distance: float) -> float:
    """Smallest slider value that includes `distance` (llmware keeps matches strictly below the cutoff)."""
    steps = math.floor(distance / DISTANCE_SLIDER_STEP + 1e-9) + 1
    return round(steps * DISTANCE_SLIDER_STEP, 2)


def set_distance_cutoff(safe_lib: str, value: float) -> None:
    """Keep both cutoff sliders in step and save the value for this corpus."""
    value = round(min(max(float(value), DISTANCE_SLIDER_MIN), DISTANCE_SLIDER_MAX), 2)
    st.session_state[f"distance_threshold_{safe_lib}"] = value
    st.session_state[f"search_distance_threshold_{safe_lib}"] = value
    save_corpus_config(safe_lib, {"distance_threshold": value})


def sync_cutoff_from_sidebar(safe_lib: str) -> None:
    # The sidebar slider is saved with "Save corpus settings"; only mirror it here.
    st.session_state[f"search_distance_threshold_{safe_lib}"] = st.session_state[f"distance_threshold_{safe_lib}"]


def sync_cutoff_from_search(safe_lib: str) -> None:
    set_distance_cutoff(safe_lib, st.session_state[f"search_distance_threshold_{safe_lib}"])


def render_distance_map(threshold: float, distances: list[float]) -> None:
    """Coloured bar under the cutoff slider: green = included, amber = raise the cutoff into this band."""
    span = DISTANCE_SLIDER_MAX - DISTANCE_SLIDER_MIN

    def pct(value: float) -> float:
        return min(max((value - DISTANCE_SLIDER_MIN) / span, 0.0), 1.0) * 100

    nearest = min(distances)
    farthest = max(distances)
    segments = [f'<div style="position:absolute;left:0;width:{pct(threshold):.1f}%;height:100%;'
                f'background:rgba(33,195,84,.55)"></div>']
    if farthest >= threshold:
        start = max(threshold, nearest)
        segments.append(
            f'<div style="position:absolute;left:{pct(threshold):.1f}%;'
            f'width:{pct(start) - pct(threshold):.1f}%;height:100%;background:rgba(255,164,33,.35)"></div>'
            f'<div style="position:absolute;left:{pct(start):.1f}%;'
            f'width:{max(pct(farthest) - pct(start), 1.0):.1f}%;height:100%;background:rgba(255,164,33,.85)"></div>'
        )
    ticks = "".join(
        f'<div style="position:absolute;left:{pct(d):.1f}%;top:-3px;width:2px;height:14px;'
        f'background:{"#1a7f37" if d < threshold else "#d1242f"}"></div>'
        for d in distances
    )
    st.markdown(
        '<div style="position:relative;height:8px;margin:-0.6rem 0.5rem 0.2rem;border-radius:4px;'
        f'background:rgba(128,128,128,.22)">{"".join(segments)}{ticks}</div>',
        unsafe_allow_html=True,
    )

    if nearest >= threshold:
        target = cutoff_to_include(nearest)
        if target <= DISTANCE_SLIDER_MAX:
            st.caption(
                f"Closest match **{nearest:.3f}** is above the cutoff. "
                f"Raise to **{target:.2f}** (amber band) to include it."
            )
        else:
            st.caption(f"Closest match **{nearest:.3f}** is beyond the slider range.")
    elif farthest >= threshold:
        st.caption(f"Raise to **{min(cutoff_to_include(farthest), DISTANCE_SLIDER_MAX):.2f}** to include all {len(distances)} passages.")
    else:
        st.caption(f"All {len(distances)} passages are within the cutoff ({nearest:.3f}–{farthest:.3f}).")


def persist_question() -> None:
    save_app_settings({"last_question": str(st.session_state.get("question_text", ""))})


def persist_search_all_libraries_setting() -> None:
    save_app_settings({"search_all_libraries": bool(st.session_state.get("search_all_libraries", False))})


def get_search_filter_from_widgets() -> dict:
    return {
        field: clean_filter_terms(st.session_state.get(f"filter_{field}", []))
        for field in SEARCH_FILTER_FIELDS
    }


def persist_search_filter() -> None:
    """Save the filter; once every field is empty there is no filtered set, so exclusions are reset."""
    current = get_search_filter_from_widgets()
    if not is_filter_active(current):
        clear_excluded_notes()
    save_app_settings({"search_filter": current})


def search_inputs() -> tuple[str, str]:
    """What's typed to search with: the question and Find notes about."""
    return st.session_state.get("question_text", ""), st.session_state.get("find_notes_query", "")


def remember_search_inputs() -> None:
    """Note what's typed, once the question box is drawn, so a rerun with the same text isn't a new search."""
    st.session_state["filter_revealed_for"] = search_inputs()


def reveal_filter_for_new_search(terms_by_field: dict[str, list[str]]) -> None:
    """Open Narrow by terms, and the folded fields holding terms, when a new search starts with a filter set.

    A new search is a changed question or Find notes about text, or a Search or Find notes click (read from
    their keys, as the buttons come after the filter). In between, the boxes stay as you left them.
    """
    clicked = st.session_state.get("search_button") or st.session_state.get("find_notes_button")
    if any(terms_by_field.values()) and (clicked or search_inputs() != st.session_state.get("filter_revealed_for")):
        st.session_state["narrow_by_terms"] = True
        for field, terms in terms_by_field.items():
            if field != "all" and terms:
                st.session_state[f"filter_fold_{field}"] = True


# Quick date buttons under Must include. Notes carry their date as 2026-10-25, and weekly notes are headed
# "Week#43"; ISO weeks (Monday to Sunday) and ISO week-years, so 29 Dec 2025 is in 2026 Week#1.
QUICK_DATE_LABELS = {
    "today": "Today", "yesterday": "-1 day", "this_week": "This wk", "last_week": "-1 wk",
    "this_month": "This mnth", "last_month": "-1 mnth",
}
# Any date-like term, typed or added by a button: 2026-10-25, 2026-10, 2026, 2026/10/25, 2026/10, 25/10/2026,
# 25-10-2026, Week#43 / Week 43. A quick date button removes the ones left over from the last search.
QUICK_DATE_TERM = re.compile(
    r"^(\d{4}([-/]\d{1,2}){0,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|week\s*#?\s*\d{1,2})$", re.IGNORECASE
)


def quick_date_terms(kind: str, today: datetime.date, field: str = "all") -> list[str]:
    """Terms for a quick date button in a filter field, e.g. ["2026-10-25"] or ["2026-10"].

    A week in Must include (AND) is the year plus "Week#43", as weekly notes are headed. In Any of (OR) and
    Exclude (NOT) a bare year would match or drop every note that year, so a week there is its seven dates.
    """
    if kind in ("today", "yesterday"):
        day = today - datetime.timedelta(days=0 if kind == "today" else 1)
        return [day.isoformat()]
    if kind in ("this_month", "last_month"):
        month_start = today.replace(day=1)
        month = month_start if kind == "this_month" else (month_start - datetime.timedelta(days=1))
        return [f"{month:%Y-%m}"]  # matches every 2026-10-DD date in that month
    day = today - datetime.timedelta(weeks=0 if kind == "this_week" else 1)
    year, week, weekday = day.isocalendar()
    if field == "all":
        return [str(year), f"Week#{week}"]
    monday = day - datetime.timedelta(days=weekday - 1)
    return [(monday + datetime.timedelta(days=i)).isoformat() for i in range(7)]


def date_terms(terms) -> list[str]:
    return [str(t) for t in terms if QUICK_DATE_TERM.match(str(t))]


def remember_searched_dates() -> None:
    """Note each field's dates at a search (or as restored at start); later clicks treat them as left over."""
    for field in SEARCH_FILTER_FIELDS:
        st.session_state[f"searched_dates_{field}"] = date_terms(st.session_state.get(f"filter_{field}", []))


def apply_quick_date(kind: str, field: str) -> None:
    """Add this button's dates to a filter field, then save.

    Dates left over from the last search (typed or from a button) are taken out first; dates added since,
    such as Today then Yesterday, are kept so they can be searched together.
    """
    key = f"filter_{field}"
    stale = set(st.session_state.get(f"searched_dates_{field}", []))
    kept = [t for t in st.session_state.get(key, []) if str(t) not in stale]
    st.session_state[key] = kept + [t for t in quick_date_terms(kind, datetime.date.today(), field) if t not in kept]
    st.session_state[f"searched_dates_{field}"] = []
    persist_search_filter()


QUICK_SEARCH_TERM_SLOTS = 5
# Each route has its colour: the Filter box takes the selected route's, and results the colour of the route that
# found them (gathered notes the route that found them last).
ROUTE_TINTS = {"semantic": "28, 131, 225", "literal": "255, 140, 0"}  # blue, orange
LLM_TINT = "33, 195, 84"  # light green: the Ask box and the LLM's answer
AGENT_TINT = "28, 131, 225"  # a harder blue for an Agent's workflow and actions


# Coloured sections set --section to their colour ("r, g, b"); inputs and buttons inside take shades of it.
SHADED_SECTIONS = (
    ".st-key-filter_box", ".st-key-ask_box", '[class*="st-key-results_"]', ".st-key-agent_panel", ".st-key-agent_prompts",
)


def section_shading_css() -> str:
    """Inputs in a coloured section get a light tint of its colour with a mid-tone border; buttons a deeper tint,
    deeper again on hover. Text keeps the theme's colour, so it reads in light and dark mode. A section inside
    another (an Agent box in the answer) uses its own colour, as --section is inherited from the nearest one."""
    within = f":is({', '.join(SHADED_SECTIONS)})"
    fields = (f'{within} :is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"], '
              '[data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] [role="group"], '
              '[data-testid="stMultiSelect"] [role="group"], [data-testid="stDateInputField"])')
    inner = f'{within} :is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"]) *'
    buttons = (f'{within} :is(button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-primary"], '
               'a[data-testid="stBaseLinkButton-secondary"])')
    return (
        f"{fields} {{ background-color: rgba(var(--section), 0.10) !important; "
        "border-color: rgba(var(--section), 0.45) !important; }"
        f"{inner} {{ background-color: transparent !important; }}"
        f"{buttons} {{ background-color: rgba(var(--section), 0.24) !important; "
        "border-color: rgba(var(--section), 0.55) !important; color: inherit !important; }"
        f"{buttons}:hover {{ background-color: rgba(var(--section), 0.36) !important; "
        "border-color: rgb(var(--section)) !important; }"
    )


def route_tint_style(route: str) -> str:
    rows = "".join(
        f'[class*="st-key-gathered_{name}_"] {{ background: rgba({rgb}, 0.10); border-radius: 0.5rem; }}'
        for name, rgb in ROUTE_TINTS.items()
    )
    # The LLM side in green (the Ask box and the answer), an Agent's workflow inside it in a harder blue.
    llm = (f".st-key-filter_box {{ --section: {ROUTE_TINTS[route]}; }}"
           f'.st-key-ask_box, [class*="st-key-results_"] {{ --section: {LLM_TINT}; }}'
           f".st-key-agent_panel {{ --section: {AGENT_TINT}; }}"
           f'.st-key-ask_box, [class*="st-key-results_"] {{ background: rgba({LLM_TINT}, 0.10) !important; }}'
           '[class*="st-key-results_"] { padding: 0.5rem 0.9rem; border-radius: 0.75rem; }'
           f".st-key-agent_panel {{ background: rgba({AGENT_TINT}, 0.16) !important; }}"
           f".st-key-agent_staging {{ background: rgba({AGENT_TINT}, 0.10) !important; }}")
    # The route switch: each option in its colour, lightly when not chosen and a few shades deeper when it is.
    switch = "".join(
        f'.st-key-route_switch button[role="radio"]:nth-of-type({n}) {{ background: rgba({rgb}, 0.10) !important; '
        f"color: inherit !important; border-color: rgba({rgb}, 0.6) !important; }}"  # theme text: readable (AA)
        f'.st-key-route_switch button[role="radio"]:nth-of-type({n})[aria-checked="true"] {{ '
        f"background: rgba({rgb}, 0.34) !important; color: inherit !important; "
        f"border-color: rgb({rgb}) !important; font-weight: 600; }}"
        f'.st-key-route_switch button[role="radio"]:nth-of-type({n}):hover {{ border-color: rgb({rgb}) !important; }}'
        for n, rgb in enumerate(ROUTE_TINTS.values(), 1)  # same order as SEARCH_ROUTES: semantic, literal
    )
    return (f"<style>.st-key-filter_box {{ background: rgba({ROUTE_TINTS[route]}, 0.06) !important; }}"
            f'[class*="st-key-gathered_"] {{ padding: 0.1rem 0.4rem; }}{rows}{llm}{section_shading_css()}{switch}</style>')


# Filter fields as small rounded boxes, and their quick dates and quick terms as rounded chips.
FILTER_FIELD_STYLE = """<style>
.st-key-filter_box .st-key-clear_all button[data-testid] { border: 1px solid #e5484d !important; color: #e5484d !important; }
.st-key-filter_box .st-key-clear_all button[data-testid]:hover {
  border-color: #e5484d !important; background: color-mix(in srgb, #e5484d 14%, transparent) !important;
}
[class*="st-key-filter_field_"] { padding: 0.45rem 0.7rem 0.6rem !important; border-radius: 0.75rem !important; }
[class*="st-key-quick_date_"] button, [class*="st-key-quick_term_"] button {
  border: 1px solid color-mix(in srgb, currentColor 28%, transparent) !important;
  border-radius: 999px !important; padding: 0 0.45rem !important; min-height: 1.5rem !important;
}
[class*="st-key-quick_date_"] button p, [class*="st-key-quick_term_"] button p { font-size: 0.8rem; }
[class*="st-key-quick_date_"] button:hover, [class*="st-key-quick_term_"] button:hover {
  border-color: currentColor !important;
}
/* Max results, the filter in words, exclusions and Clear exclusions stay on one line; the filter text takes
   what's left and wraps within it. */
.st-key-max_results_row { flex-wrap: nowrap; }
.st-key-max_results_row > div { min-width: 0; }
.st-key-max_results_row > div:has([data-testid="stCaptionContainer"]) { flex-shrink: 1; }
/* The folded OR and NOT fields: a tighter fold, like the Must include box. */
[class*="st-key-filter_fold_"] summary { padding-top: 0.35rem !important; padding-bottom: 0.35rem !important; }
</style>"""


def quick_search_terms() -> list[str]:
    """The quick search terms set on the prompt library page (up to 5, blanks dropped)."""
    terms = load_prompt_library().get("quick_search_terms") or []
    return [str(t).strip() for t in terms[:QUICK_SEARCH_TERM_SLOTS] if str(t).strip()]


def save_quick_search_terms() -> None:
    library = load_prompt_library()
    library["quick_search_terms"] = [
        str(st.session_state.get(f"quick_search_term_{i}", "")).strip() for i in range(QUICK_SEARCH_TERM_SLOTS)
    ]
    save_prompt_library(library)


def apply_quick_term(term: str, field: str) -> None:
    """Add a quick search term to a filter field (once), then save."""
    key = f"filter_{field}"
    st.session_state[key] = clean_filter_terms(list(st.session_state.get(key, [])) + [term])
    persist_search_filter()


def render_filter_heading(field: str, label: str, field_help: str, show_label: bool = True) -> None:
    """A filter field's heading with its quick date buttons on the same line, and quick search terms below.

    In a fold the fold's title names the field, so `show_label=False` leaves just the help icon.
    """
    today = datetime.date.today()
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
        st.markdown(label if show_label else "", width="content", help=field_help)
        for kind, button_label in QUICK_DATE_LABELS.items():
            terms = quick_date_terms(kind, today, field)
            shown = " and ".join(f"`{t}`" for t in terms) if len(terms) <= 2 else f"`{terms[0]}` … `{terms[-1]}`"
            st.button(
                button_label,
                key=f"quick_date_{field}_{kind}",
                type="tertiary",
                on_click=apply_quick_date,
                args=(kind, field),
                help=f"Adds {shown} to **{label}**, and takes out dates left from the last search.",
            )
    terms = quick_search_terms()
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
        for i, term in enumerate(terms):
            st.button(
                term,
                key=f"quick_term_{field}_{i}",
                type="tertiary",
                on_click=apply_quick_term,
                args=(term, field),
                help=f"Adds `{term}` to **{label}**.",
            )
        st.link_button(
            ":material/edit:" if terms else "Set quick search terms",
            f"{PROMPT_LIBRARY_PAGE_URL}&section=quick_search",
            type="tertiary",
            help="Set up to 5 quick search terms in the prompt library (opens in a new tab).",
        )


def clear_search_filter() -> None:
    """Clear Filter button: empty every filter field and drop note exclusions and gathered notes."""
    for field in SEARCH_FILTER_FIELDS:
        st.session_state[f"filter_{field}"] = []
    save_app_settings({"search_filter": {field: [] for field in SEARCH_FILTER_FIELDS}})
    clear_excluded_notes()
    clear_gathered_notes()


def switch_to_library(library: str) -> None:
    """Button callback: make `library` the selected library (runs before widgets are created)."""
    st.session_state["selected_library"] = library
    save_app_settings({"last_selected_library": library})

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_embedding_model_catalog() -> tuple[dict, ...]:
    """Return local embedding models available in the LLMWare catalog."""
    models: list[dict] = []
    for card in ModelCatalog().list_embedding_models():
        if card.get("model_family") not in LOCAL_EMBEDDING_FAMILIES:
            continue

        model_id = card.get("display_name") or card.get("model_name")
        if not model_id:
            continue
        if any(keyword in model_id.lower() for keyword in EXCLUDED_MODEL_KEYWORDS):
            continue

        models.append(
            {
                "id": model_id,
                "label": f"{model_id} ({card.get('embedding_dims', '?')}d)",
                "dims": card.get("embedding_dims", 384),
                "recommended": model_id in RECOMMENDED_EMBEDDING_MODELS,
                # Only HF models get the GPU acceleration patch.
                "gpu": card.get("model_family") == "HFEmbeddingModel",
            }
        )

    models.sort(
        key=lambda m: (
            RECOMMENDED_EMBEDDING_MODELS.index(m["id"])
            if m["id"] in RECOMMENDED_EMBEDDING_MODELS
            else len(RECOMMENDED_EMBEDDING_MODELS) + 1,
            m["label"],
        )
    )
    return tuple(models)


def get_model_dims(embedding_model: str) -> int:
    card = ModelCatalog().lookup_model_card(embedding_model)
    return int(card.get("embedding_dims", 384)) if card else 384


def get_embedding_batch_size(embedding_model: str) -> int:
    """Large HF models embed slowly — use smaller batches to avoid stalls/OOM."""
    if embedding_model in {"gte-large", "gte-large-en-v1.5", "bge-large-en-v1.5"}:
        return 16
    if "large" in embedding_model.lower():
        return 32
    return 100


def get_embedding_extra_deps(embedding_model: str) -> tuple[str, ...]:
    """Return optional pip packages needed for a given embedding model."""
    return EMBEDDING_MODEL_EXTRA_DEPS.get(embedding_model, ())


def missing_python_packages(packages: tuple[str, ...] | list[str]) -> list[str]:
    """Return import names from *packages* that are not available in the current interpreter."""
    import importlib.util

    missing: list[str] = []
    for package in packages:
        if not importlib.util.find_spec(package):
            missing.append(package)
    return missing


def normalize_pip_package_name(raw: str) -> str:
    """Convert LLMWare dependency names to pip-installable package names."""
    pkg = re.split(r"[<>=!]", raw, maxsplit=1)[0].strip()
    return PIP_PACKAGE_ALIASES.get(pkg, pkg)


def parse_pip_packages_from_error(exc: BaseException) -> list[str]:
    """Extract pip package names from LLMWare / transformers install errors."""
    packages: list[str] = []

    try:
        from llmware.configs import DependencyNotInstalledException

        if isinstance(exc, DependencyNotInstalledException):
            match = re.search(r"'([^']+)' needs to be installed", str(exc))
            if match:
                pkg = normalize_pip_package_name(match.group(1))
                if pkg:
                    return [pkg]
    except ImportError:
        pass

    message = str(exc)
    for match in re.finditer(r"pip install\s+`?([a-zA-Z0-9_.-]+)`?", message, flags=re.IGNORECASE):
        pkg = normalize_pip_package_name(match.group(1))
        if pkg and pkg not in packages:
            packages.append(pkg)
    if packages:
        return packages

    needs_install = re.search(
        r"'([^']+)'\s+needs to be installed",
        message,
        flags=re.IGNORECASE,
    )
    if needs_install:
        pkg = normalize_pip_package_name(needs_install.group(1))
        if pkg:
            packages.append(pkg)
        return packages

    lowered = message.lower()
    if "no module named" in lowered:
        module_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", message, flags=re.IGNORECASE)
        if module_match:
            pkg = normalize_pip_package_name(module_match.group(1).split(".")[0])
            if pkg and pkg not in packages:
                packages.append(pkg)
    return packages


def resolve_pip_install_context() -> tuple[str, list[str]]:
    """Return (installer_label, shell command lines) for dependency install instructions."""
    venv_dir = PROJECT_DIR / "venv"
    in_venv = sys.prefix != sys.base_prefix

    lines = [f"cd {PROJECT_DIR}"]
    if venv_dir.is_dir():
        lines.append(f"source {venv_dir}/bin/activate")
        return "pip (project venv)", lines
    if in_venv:
        return "pip (active venv)", lines
    return "pip (system Python)", lines


def format_dependency_install_command(packages: list[str]) -> str:
    """Build copy-paste shell commands to install missing optional dependencies."""
    if not packages:
        return ""

    _installer, lines = resolve_pip_install_context()
    lines.append(f"pip install {' '.join(packages)}")
    for pkg in packages:
        for extra in OPTIONAL_DEPENDENCY_EXTRA_LINES.get(pkg, []):
            lines.append(extra)
    return "\n".join(lines)


def format_embedding_dependency_install_command(
    packages: list[str],
    embedding_model: str | None = None,
) -> str:
    return format_dependency_install_command(packages)


def dependency_relaunch_hint() -> str:
    """Plain-text hint shown after the install commands."""
    return (
        f"After installing, relaunch the app using the LLM-Notes Desktop shortcut, "
        f"or from Terminal run: cd {PROJECT_DIR} && ./start_app.sh"
    )


def embedding_dependency_relaunch_hint() -> str:
    return dependency_relaunch_hint()


class DependencyInstallError(RuntimeError):
    """Raised when a scan needs optional packages that are not installed."""

    def __init__(self, packages: list[str], context: str = ""):
        self.packages = packages
        self.context = context
        super().__init__(context or ", ".join(packages))


class EmbeddingDependencyError(DependencyInstallError):
    """Raised when an embedding model needs optional packages that are not installed."""

    def __init__(self, embedding_model: str, packages: list[str]):
        self.embedding_model = embedding_model
        super().__init__(packages, embedding_model)


def render_dependency_install_help(
    packages: list[str],
    *,
    sidebar: bool = False,
    intro: str | None = None,
) -> None:
    """Show install commands in a copy-paste-friendly code block."""
    write = st.sidebar if sidebar else st
    pkg_text = ", ".join(f"`{pkg}`" for pkg in packages)
    if intro is None:
        intro = f"Missing package(s): {pkg_text}. Copy and run in Terminal:"
    write.warning(intro)
    write.code(format_dependency_install_command(packages), language="bash")
    write.caption(dependency_relaunch_hint())


def render_embedding_dependency_install_help(
    packages: list[str],
    embedding_model: str,
    *,
    sidebar: bool = False,
    intro: str | None = None,
) -> None:
    if intro is None:
        intro = (
            f"`{embedding_model}` needs extra package(s): "
            f"{', '.join(f'`{p}`' for p in packages)}. "
            "Copy and run in Terminal:"
        )
    render_dependency_install_help(packages, sidebar=sidebar, intro=intro)


def get_missing_embedding_dependencies(embedding_model: str) -> list[str]:
    """Return pip packages missing for *embedding_model* (known extras only)."""
    return missing_python_packages(get_embedding_extra_deps(embedding_model))


def ensure_embedding_dependencies(embedding_model: str) -> None:
    """Raise EmbeddingDependencyError when model extras are not installed."""
    missing = get_missing_embedding_dependencies(embedding_model)
    if missing:
        raise EmbeddingDependencyError(embedding_model, missing)


def format_exception_for_user(exc: BaseException) -> str:
    """Turn low-level parser/DB exceptions into readable scan errors."""
    try:
        from pytesseract.pytesseract import TesseractError

        if isinstance(exc, TesseractError):
            return (
                "Tesseract OCR failed on a note image (corrupt or unsupported format). "
                "The app now skips bad images automatically — try **Rescan** again."
            )
    except ImportError:
        pass

    if isinstance(exc, sqlite3.Error):
        message = str(exc).lower()
        if "unable to open database file" in message:
            return (
                "SQLite could not open the index database — usually **too many open files** "
                "during a large full-corpus scan. Quit and relaunch the app (Desktop shortcut "
                "or `./start_app.sh`), then run **Re-index** again. "
                "If it repeats, close other apps and run in Terminal first: `ulimit -n 65536` "
                "then `./start_app.sh`."
            )
        return f"Database error during ingest: {exc}"

    message = str(exc).strip()
    if message.startswith("(") and "Error during processing" in message:
        return (
            "Tesseract OCR failed on a note image (corrupt or unsupported format). "
            "The app now skips bad images automatically — try **Rescan** again."
        )
    return message


def handle_scan_run_error(
    exc: BaseException,
    embedding_model: str | None = None,
    *,
    sidebar: bool = False,
) -> None:
    """Show actionable install help for dependency failures during ingest or embedding."""
    if isinstance(exc, EmbeddingDependencyError):
        render_embedding_dependency_install_help(
            exc.packages, exc.embedding_model, sidebar=sidebar
        )
        return
    if isinstance(exc, DependencyInstallError):
        render_dependency_install_help(exc.packages, sidebar=sidebar)
        return

    missing = parse_pip_packages_from_error(exc)
    if missing:
        pkg_text = ", ".join(f"`{pkg}`" for pkg in missing)
        if embedding_model and any(pkg in get_embedding_extra_deps(embedding_model) for pkg in missing):
            render_embedding_dependency_install_help(missing, embedding_model, sidebar=sidebar)
        else:
            render_dependency_install_help(
                missing,
                sidebar=sidebar,
                intro=(
                    f"Ingesting notes needs package(s): {pkg_text}. "
                    "Copy and run in Terminal, then rescan:"
                ),
            )
        return

    write = st.sidebar if sidebar else st
    write.error(format_exception_for_user(exc))


def handle_embedding_run_error(exc: BaseException, embedding_model: str, *, sidebar: bool = False) -> None:
    handle_scan_run_error(exc, embedding_model, sidebar=sidebar)


def lancedb_table_names(db) -> list[str]:
    """Return LanceDB table names across API versions."""
    if hasattr(db, "list_tables"):
        listed = db.list_tables()
        if hasattr(listed, "tables"):
            return list(listed.tables)
        return list(listed)
    return list(db.table_names())


def get_model_label(embedding_model: str) -> str:
    for model in load_embedding_model_catalog():
        if model["id"] == embedding_model:
            return model["label"]
    return embedding_model


def get_embedding_model_summary(embedding_model: str) -> str:
    """Return a short human-readable summary for an embedding model."""
    if embedding_model in EMBEDDING_MODEL_SUMMARIES:
        return EMBEDDING_MODEL_SUMMARIES[embedding_model]

    lowered = embedding_model.lower()
    if "industry-bert" in lowered:
        return "Domain-specific embedding model for specialized document collections."
    if "large" in lowered:
        return "Higher-quality embedding model. Slower to index, better retrieval."
    if "small" in lowered or "mini" in lowered:
        return "Compact embedding model. Faster indexing, moderate retrieval quality."

    return "Local embedding model used to power semantic search."


def will_embed_on_gpu(embedding_model: str) -> bool:
    """True when this model would embed on a GPU on this machine (whether or not the GPU option is ticked)."""
    if _pick_embedding_device(True) == "cpu":
        return False
    return any(m["id"] == embedding_model and m["gpu"] for m in load_embedding_model_catalog())


def format_embedding_model_option(model_id: str, model_labels: dict[str, str]) -> str:
    """Format a selectbox label with model name, size, and summary; (g) marks models that will use the GPU."""
    label = model_labels.get(model_id, model_id)
    summary = get_embedding_model_summary(model_id)
    prefix = "(g) " if will_embed_on_gpu(model_id) else ""
    return f"{prefix}{label} — {summary}"


def load_all_corpus_settings() -> dict:
    if not CORPUS_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(CORPUS_SETTINGS_PATH.read_text())
    except Exception:
        return {}


def get_corpus_config(safe_lib: str) -> dict:
    config = DEFAULT_CORPUS_CONFIG.copy()
    config.update(load_all_corpus_settings().get(safe_lib, {}))
    return config


def save_corpus_config(safe_lib: str, updates: dict) -> dict:
    all_settings = load_all_corpus_settings()
    config = get_corpus_config(safe_lib)
    config.update(updates)
    all_settings[safe_lib] = config
    CORPUS_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_SETTINGS_PATH.write_text(json.dumps(all_settings, indent=2))
    return config


def get_active_corpus_prompt(corpus_config: dict) -> str:
    """Corpus prompt to send to the LLM, or "" when it is empty or not applied."""
    if not corpus_config.get("corpus_prompt_enabled"):
        return ""
    return str(corpus_config.get("corpus_prompt", "")).strip()


def persist_corpus_prompt(safe_lib: str) -> None:
    save_corpus_config(
        safe_lib,
        {
            "corpus_prompt": str(st.session_state.get(f"corpus_prompt_{safe_lib}", "")),
            "corpus_prompt_enabled": bool(st.session_state.get(f"corpus_prompt_enabled_{safe_lib}", False)),
        },
    )


def mark_text_saved(field_key: str) -> None:
    """Remember when a text box was last saved, for the 'Saved …' note under it."""
    st.session_state.setdefault("text_saved_at", {})[field_key] = datetime.datetime.now()


def render_saved_note(field_key: str) -> None:
    saved_at = st.session_state.get("text_saved_at", {}).get(field_key)
    if saved_at:
        st.caption(f":material/check: Saved {saved_at:%H:%M:%S} on {saved_at:%d %b %Y}")


def persist_corpus_prompt_text(safe_lib: str) -> None:
    persist_corpus_prompt(safe_lib)
    mark_text_saved(f"corpus_prompt_{safe_lib}")


def render_corpus_prompt_editor(safe_lib: str, corpus_config: dict) -> None:
    """Editor for the per-corpus author-role prompt (shown on the Prompt library page)."""
    st.checkbox(
        "Apply corpus prompt",
        value=bool(corpus_config.get("corpus_prompt_enabled", False)),
        key=f"corpus_prompt_enabled_{safe_lib}",
        on_change=persist_corpus_prompt,
        args=(safe_lib,),
        help=(
            "When ticked, this text is sent to the LLM ahead of the notes for every answer "
            "and follow-up in this corpus. Saved per corpus."
        ),
    )
    st.text_area(
        "Your role as the author",
        value=str(corpus_config.get("corpus_prompt", "")),
        key=f"corpus_prompt_{safe_lib}",
        on_change=persist_corpus_prompt_text,
        args=(safe_lib,),
        height=120,
        placeholder="e.g. I wrote these notes as the owner of a small consultancy, tracking clients and proposals.",
        help="Describe who you were when writing these notes. Press Cmd+Enter or click away to save.",
    )
    render_saved_note(f"corpus_prompt_{safe_lib}")


# ---------------------------------------------------------------------------
# PROMPT LIBRARY — About me, global prompts and task prompts for the LLM
# ---------------------------------------------------------------------------

DEFAULT_TASK_PROMPTS = (
    {
        "name": "Meeting summary email",
        "text": (
            "Write a meeting summary and next steps as an email I can send to the attendees. "
            "Open with one line on the purpose of the meeting, then the key points as short bullets, "
            "the decisions made, and the next steps with owners and dates where the notes give them. "
            "Keep it under 250 words."
        ),
        "tags": ["email", "meeting"],
        "starred": True,
        "agent": True,
        "output": "email",
    },
    {
        "name": "Action items",
        "text": (
            "List every action item in these notes as a checklist. Give the owner and due date when "
            "the notes say, and flag anything overdue. Don't invent owners or dates."
        ),
        "tags": ["tasks"],
        "starred": True,
        "agent": True,
        "output": "task",
    },
    {
        "name": "Client status update",
        "text": (
            "Draft a short status update for this client: where things stand, what changed recently, "
            "risks or blockers, and the next step. Plain language, no more than 150 words."
        ),
        "tags": ["client", "update"],
        "agent": True,
        "output": "note",
    },
    {
        "name": "Key decisions",
        "text": "Summarise the decisions recorded in these notes, with the date and who made each one, oldest first.",
        "tags": ["summary"],
    },
)


def new_prompt_record(
    name: str,
    text: str,
    *,
    global_prompt: bool = False,
    tags=None,
    starred: bool = False,
    agent: bool = False,
    output: str = "text",
    model: str = "",
    references=None,
    salesforce: bool = False,
) -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "name": name.strip(),
        "text": text.strip(),
        "global": bool(global_prompt),
        "agent": bool(agent),
        "output": output,
        "steps": [],  # Agent workflow steps between the collection prompt (text) and the command
        "command": DEFAULT_AGENT_COMMANDS.get(output, "") if agent else "",
        "model": model,  # empty: use whichever model is selected
        "references": list(references or []),  # reference documents sent whenever this prompt is used
        "salesforce": bool(salesforce),  # send the Salesforce records the notes mention whenever it's used
        "tags": clean_filter_terms(tags or []),
        "starred": bool(starred),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "last_used": None,
    }


def save_prompt_library(library: dict) -> dict:
    PROMPT_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_LIBRARY_PATH.write_text(json.dumps(library, indent=2))
    return library


def load_prompt_library() -> dict:
    """Read the prompt library, seeding example task prompts the first time."""
    if PROMPT_LIBRARY_PATH.exists():
        try:
            library = json.loads(PROMPT_LIBRARY_PATH.read_text())
        except Exception:
            library = {}
        library.setdefault("profile", {"text": "", "enabled": True})
        library.setdefault("prompts", [])
        library.setdefault("links", [])  # common links: [{"id", "title", "url"}]; prompts pick theirs by id
        library.setdefault("references", [])  # reference documents: [{"id", "title", "notes", "last_used"?}]
        library.setdefault("semantic_fixes", [])  # Agent Semantic Fixes: [{"id", "text", "added", "edited"?}]
        for prompt in library["prompts"]:
            # Libraries saved before Agent existed: prompts that already drafted output become Agents.
            prompt.setdefault("agent", prompt.get("output", "text") in DRAFT_OUTPUT_TYPES and not prompt.get("global"))
            # ...and before Agent workflows: no middle steps, a default Action Command, no model of its own.
            prompt.setdefault("steps", [])
            prompt.setdefault("command", DEFAULT_AGENT_COMMANDS.get(prompt.get("output", ""), ""))
            prompt.setdefault("model", "")
            prompt.setdefault("links", [])
            prompt.setdefault("references", [])  # reference documents sent whenever this prompt is used
            prompt.setdefault("salesforce", False)
        return library
    return save_prompt_library(
        {
            "profile": {"text": "", "enabled": True},
            "prompts": [
                new_prompt_record(
                    p["name"],
                    p["text"],
                    tags=p.get("tags"),
                    starred=p.get("starred", False),
                    agent=p.get("agent", False),
                    output=p.get("output", "text"),
                )
                for p in DEFAULT_TASK_PROMPTS
            ],
        }
    )


def find_prompt(library: dict, prompt_id: str | None) -> dict | None:
    return next((p for p in library["prompts"] if p["id"] == prompt_id), None)


def update_prompt(prompt_id: str, **fields) -> None:
    library = load_prompt_library()
    prompt = find_prompt(library, prompt_id)
    if prompt:
        prompt.update(fields)
        save_prompt_library(library)


def delete_prompt(prompt_id: str) -> None:
    library = load_prompt_library()
    library["prompts"] = [p for p in library["prompts"] if p["id"] != prompt_id]
    save_prompt_library(library)
    remove_active_task_prompt(prompt_id)


def mark_prompt_used(prompt_id: str) -> None:
    update_prompt(prompt_id, last_used=datetime.datetime.now().isoformat(timespec="seconds"))


def select_prompt_stacks(library: dict) -> tuple[list[dict], list[dict]]:
    """Five most recently used task prompts, and every starred one, most recently used first.

    The Ask panel leaves out of its starred rows only the prompts its Recent row actually shows, so a
    starred prompt never vanishes because Recent ran out of room.
    """
    task_prompts = [p for p in library["prompts"] if not p.get("global")]
    recent = sorted(
        (p for p in task_prompts if p.get("last_used")), key=lambda p: p["last_used"], reverse=True
    )[:PROMPT_STACK_SIZE]
    starred = sorted(
        (p for p in task_prompts if p.get("starred")),
        key=lambda p: (p.get("last_used") or "", p.get("created") or ""),
        reverse=True,
    )
    return recent, starred


def prompt_matches_query(prompt: dict, query: str) -> bool:
    query = query.strip().lower()
    if not query:
        return True
    haystack = " ".join([prompt.get("name", ""), prompt.get("text", ""), *prompt.get("tags", [])]).lower()
    return all(word in haystack for word in query.split())


def truncate_text(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_prompt_context(corpus_prompt: str = "", profile_text: str = "", global_texts=(), references=()) -> str:
    """Preamble sent ahead of the notes: today's date, About me, corpus prompt, global prompts, reference documents.

    `references` is [(short title, text)].
    """
    parts = [f"Today's date: {datetime.date.today():%A %d %B %Y}."]
    if profile_text.strip():
        parts.append(f"About the person asking (the owner of these notes):\n{profile_text.strip()}")
    if corpus_prompt.strip():
        parts.append(
            "Context about the author of these notes:\n"
            f"{corpus_prompt.strip()}\n"
            "Use this to interpret the notes and frame your answer."
        )
    standing = [text.strip() for text in global_texts if text.strip()]
    if standing:
        parts.append("Always follow these instructions:\n" + "\n".join(f"- {text}" for text in standing))
    documents = [(title, text) for title, text in references if text.strip()]
    if documents:
        parts.append(
            "Reference documents (background to use when answering; they are not the notes being asked about):\n\n"
            + "\n\n".join(f"=== {title} ===\n{text.strip()}" for title, text in documents)
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# HANDOFF — structured drafts written to an outbox for another system to act on
# ---------------------------------------------------------------------------

OUTPUT_TYPES = {
    "text": "Text answer", "note": "Note", "task": "Tasks", "email": "Email", "next_step": "Next step + chase task",
}
DRAFT_OUTPUT_TYPES = [t for t in OUTPUT_TYPES if t != "text"]  # what an Agent prompt can send to the outbox
DEFAULT_AGENT_OUTPUT = "task"
DEFAULT_AGENT_COMMANDS = {
    "task": "Create these as tasks", "note": "Save this as a note", "email": "Draft this as an email",
    "next_step": "Update the Opportunity's Next Step and create a task to chase",
}
AGENT_BORDER_COLOR = "#1c83e1"
HANDOFF_SCHEMA_ID = "llm-notes-handoff/v1"
HANDOFF_SCHEMA_V2_ID = "llm-notes-handoff/v2"  # adds update items and mixed packages (e.g. Next Step + chase task)
HANDOFF_SCHEMA_PATHS = {
    HANDOFF_SCHEMA_ID: Path(__file__).resolve().parent / "schemas" / "handoff.v1.schema.json",
    HANDOFF_SCHEMA_V2_ID: Path(__file__).resolve().parent / "schemas" / "handoff.v2.schema.json",
}
NEXT_STEP_LIMIT = 255  # characters an opportunity's Next Step can hold
DEFAULT_OUTBOX_DIR = PROJECT_DIR / "outbox"
HANDOFF_RESULT_DIRS = ("processed", "failed")  # where connectors move packages, with a receipt beside each
RELATED_KINDS = ["", "opportunity", "account", "contact"]
TASK_PRIORITIES = ["", "low", "normal", "high"]
STRUCTURED_OUTPUT_GUIDES = {
    "note": "Return JSON with a one-paragraph `summary` and `items` holding exactly one note: a short `title` and the note `body`.",
    "task": (
        "Return JSON with a one-paragraph `summary` and `items` holding one entry per task: a short `subject`, "
        "a `description`, `due_date` as YYYY-MM-DD worked out from today's date (or \"\" if the notes give none), "
        "`priority` (low, normal, high or \"\") and the `owner` named in the notes (or \"\")."
    ),
    "email": (
        "Return JSON with a one-paragraph `summary` and `items` holding exactly one email: `to` (names or "
        "addresses given in the notes — never invent addresses), `subject` and `body`."
    ),
    "next_step": (
        "Return JSON with a one-paragraph `summary` and `items` holding one entry per opportunity to update: "
        "`next_step` (one short sentence on where things stand and what happens next, under 200 characters; "
        "no date or initials, they're added for you), `chase_subject` (a short subject for the follow-up task), "
        "`chase_due_date` as YYYY-MM-DD worked out from today's date (or \"\" if the notes give none) and "
        "`chase_description` (or \"\"), and `already_said`: true if the opportunity's current Next Step (listed "
        "below) already says the same thing, else false."
    ),
}
STRUCTURED_RELATED_GUIDE = (
    "Set `related_to.name` to the client, deal or person the item is about when the notes name one, else \"\". "
    "Set `related_to.kind` to opportunity, account or contact when that is clear, else \"\". "
    "Use \"\" for anything the notes don't say; don't guess."
)


def is_agent_prompt(prompt: dict | None) -> bool:
    return bool(prompt and prompt.get("agent") and not prompt.get("global"))


def add_agent_border_style(*selectors: str) -> None:
    """Blue border on the elements matched by the CSS selectors (usually st-key-* classes from widget keys)."""
    st.html(f"<style>{', '.join(selectors)} {{ border: 1px solid {AGENT_BORDER_COLOR} !important; }}</style>")


def agent_prompt_label(prompt: dict) -> str:
    """Prompt name as shown in the Ask area: Agents read 'Agent:SF_Tasks'."""
    return f"Agent:{prompt['name']}" if is_agent_prompt(prompt) else prompt["name"]


def get_prompt_output_type(prompt: dict | None) -> str:
    """Only Agent prompts draft output for the outbox; every other prompt gets a text answer."""
    if not is_agent_prompt(prompt):
        return "text"
    output = prompt.get("output", "")
    return output if output in DRAFT_OUTPUT_TYPES else DEFAULT_AGENT_OUTPUT


def structured_output_schema(item_type: str) -> dict:
    """JSON schema passed to Ollama's `format` so the model can only return valid drafts."""
    related = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "kind": {"enum": RELATED_KINDS}},
        "required": ["name", "kind"],
    }
    fields = {
        "note": {"title": {"type": "string"}, "body": {"type": "string"}},
        "task": {
            "subject": {"type": "string"},
            "description": {"type": "string"},
            "due_date": {"type": "string"},
            "priority": {"enum": TASK_PRIORITIES},
            "owner": {"type": "string"},
        },
        "email": {"to": {"type": "array", "items": {"type": "string"}}, "subject": {"type": "string"}, "body": {"type": "string"}},
        "next_step": {
            "next_step": {"type": "string"},
            "chase_subject": {"type": "string"},
            "chase_due_date": {"type": "string"},
            "chase_description": {"type": "string"},
            "already_said": {"type": "boolean"},
        },
    }[item_type]
    item = {"type": "object", "properties": {**fields, "related_to": related}, "required": [*fields, "related_to"]}
    return {
        "type": "object",
        "properties": {"summary": {"type": "string"}, "items": {"type": "array", "items": item}},
        "required": ["summary", "items"],
    }


def normalise_related_to(value) -> dict:
    value = value if isinstance(value, dict) else {}
    kind = str(value.get("kind", "")).strip().lower()
    return {"name": str(value.get("name", "")).strip(), "kind": kind if kind in RELATED_KINDS else ""}


def normalise_draft_item(item_type: str, item) -> dict:
    """Coerce one model-produced item into the handoff schema shape (empty strings for unknowns)."""
    item = item if isinstance(item, dict) else {}
    text = lambda key: str(item.get(key, "") or "").strip()  # noqa: E731
    draft = {"type": item_type}
    if item_type == "note":
        draft.update(title=text("title"), body=text("body"))
    elif item_type == "update":
        draft.update(field="next_step", value=text("value"))  # next_step is the only field v2 updates
        if "previous" in item:
            draft["previous"] = str(item.get("previous") or "")
    elif item_type == "task":
        due = text("due_date")
        priority = text("priority").lower()
        draft.update(
            subject=text("subject"),
            description=text("description"),
            due_date=due if re.fullmatch(r"\d{4}-\d{2}-\d{2}", due) else "",
            priority=priority if priority in TASK_PRIORITIES else "",
            owner=text("owner"),
        )
        if item.get("allow_duplicate"):  # Create anyway, even if it looks like an open task
            draft["allow_duplicate"] = True
    else:
        recipients = item.get("to", [])
        if isinstance(recipients, str):
            recipients = recipients.split(",")
        draft.update(
            to=[str(r).strip() for r in recipients if str(r).strip()],
            subject=text("subject"),
            body=text("body"),
        )
    draft["related_to"] = normalise_related_to(item.get("related_to"))
    return draft


def agent_command(prompt: dict) -> str:
    """The Agent's Action Command, or the default for its output type."""
    return str(prompt.get("command", "")).strip() or DEFAULT_AGENT_COMMANDS[get_prompt_output_type(prompt)]


# --- Reference snapshot: current records a connector copied from the other system (read-only) -----------
# The connector writes `llm-notes-reference/v1` files to <outbox>/reference/; this app only reads them, so it
# never contacts the other system. They let the Agent use real record names and show their current state.

REFERENCE_SCHEMA_ID = "llm-notes-reference/v1"
REFERENCE_FILE = "opportunities.json"
REFERENCE_TASKS_FILE = "tasks.json"  # the user's own open tasks
OPEN_TASKS_PROMPT_LIMIT = 40  # open tasks listed to the model
REFERENCE_PROMPT_LIMIT = 80  # records listed to the model; the snapshot is newest-updated first
_reference_cache: dict = {}


def load_reference_file(name: str) -> dict | None:
    """One of the connector's snapshot files, or None if there isn't a valid one (re-read when it changes)."""
    path = get_outbox_dir() / "reference" / name
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    cached = _reference_cache.setdefault(name, {})
    if cached.get("key") != (str(path), stamp):
        try:
            snapshot = json.loads(path.read_text())
        except (OSError, ValueError):
            snapshot = None
        if not isinstance(snapshot, dict) or snapshot.get("schema") != REFERENCE_SCHEMA_ID:
            snapshot = None
        cached.update(key=(str(path), stamp), snapshot=snapshot)
    return cached["snapshot"]


def load_reference_snapshot() -> dict | None:
    """The connector's snapshot of open opportunities, or None if there isn't a valid one."""
    return load_reference_file(REFERENCE_FILE)


def reference_open_tasks() -> list[dict]:
    """The user's own open tasks from the connector's snapshot (tasks.json), soonest due first."""
    snapshot = load_reference_file(REFERENCE_TASKS_FILE) or {}
    return [r for r in snapshot.get("records", []) if isinstance(r, dict) and r.get("kind") == "task" and r.get("subject")]


# A staged task that repeats one of the user's open tasks. Same rules as the connector's sfc/duplicates.py, so
# what's flagged here is what the connector would hold: the subjects' key words, ignoring case, punctuation,
# title markers, filler, one- and two-letter words and plural s; all must match, or at least two making up 3/4
# of the shorter subject. Tasks on different records are different actions.
TASK_FILLER = {
    "the", "a", "an", "to", "for", "with", "and", "or", "of", "on", "in", "at", "by", "about", "re", "from",
    "our", "my", "your", "their", "his", "her", "its", "this", "that", "these", "those", "up", "out", "new",
    "please", "re:", "fw", "fwd",
}
TASK_MARKER = re.compile(r"\[[^\]]*\]|^[A-Z0-9_]+:\s*")


def task_subject_words(subject: str) -> set[str]:
    words = set()
    for word in re.findall(r"[a-z0-9]+", TASK_MARKER.sub(" ", subject or "").lower()):
        if len(word) <= 2 or word in TASK_FILLER:
            continue
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word)
    return words


def same_task_action(new_subject: str, old_subject: str) -> bool:
    new, old = task_subject_words(new_subject), task_subject_words(old_subject)
    if not new or not old:
        return False
    shared = len(new & old)
    return new == old or (shared >= 2 and shared / min(len(new), len(old)) >= 0.75)


def similar_open_tasks(item: dict) -> list[dict]:
    """The user's open tasks a staged task looks like (none for other item types)."""
    if item.get("type") != "task":
        return []
    on = str((item.get("related_to") or {}).get("name", "")).strip().lower()
    return [
        task for task in reference_open_tasks()
        if not (on and task.get("related") and task["related"].strip().lower() != on)
        and same_task_action(item.get("subject", ""), task["subject"])
    ]


def describe_open_task(task: dict) -> str:
    """'“Send pricing” · due 2026-10-27 · Not Started · on Acme pilot · [Open Task ↗](…)'."""
    parts = [f"“{task['subject']}”", f"due {task['due_date']}" if task.get("due_date") else "no due date",
             task.get("status") or "", f"on {task['related']}" if task.get("related") else ""]
    if str(task.get("url") or "").startswith("https://"):
        parts.append(f"[Open Task ↗]({task['url']})")
    return " · ".join(p for p in parts if p)


def open_tasks_prompt_text(prompts: list[dict], records: list[dict]) -> str:
    """The user's open tasks, for Agents that create tasks, so they don't propose one again (empty otherwise).

    Tasks on the opportunities the notes mention come first, then the soonest due, up to OPEN_TASKS_PROMPT_LIMIT.
    """
    if not any(get_prompt_output_type(p) in ("task", "next_step") for p in prompts):
        return ""
    tasks = reference_open_tasks()
    if not tasks:
        return ""
    mentioned = {str(r.get(k) or "").strip().lower() for r in records for k in ("name", "account")} - {""}
    tasks = sorted(tasks, key=lambda t: (t.get("related", "").strip().lower() not in mentioned,))[:OPEN_TASKS_PROMPT_LIMIT]
    lines = "\n".join(
        f"- {t['subject']} (" + ", ".join(p for p in (
            f"due {t['due_date']}" if t.get("due_date") else "", t.get("status", ""), f"on {t['related']}" if t.get("related") else ""
        ) if p) + ")"
        for t in tasks
    )
    return (
        "These are my open tasks already in the connected system. Don't propose a task that repeats one of them; "
        f"if an action is already covered by one, say so instead.\n{lines}"
    )


def reference_records() -> list[dict]:
    snapshot = load_reference_snapshot()
    return [r for r in (snapshot or {}).get("records", []) if isinstance(r, dict) and r.get("name")]


def find_reference_record(name: str) -> dict | None:
    """The snapshot record with exactly this name (ignoring case), as the connector matches it."""
    wanted = (name or "").strip().lower()
    return next((r for r in reference_records() if r["name"].strip().lower() == wanted), None) if wanted else None


def reference_record_link(record: dict) -> str:
    """A markdown link to the record's page in the other system, e.g. "[Open in Salesforce ↗](…)", or ""."""
    url = str(record.get("url") or "")
    if not url.startswith("https://"):
        return ""
    source = str((load_reference_snapshot() or {}).get("source") or "").strip()
    return f"[Open in {source.title() if source else 'the other system'} ↗]({url})"


# Text that must not get links inside it: existing markdown links, inline code and bare addresses.
LINK_PROTECTED = re.compile(r"(\[[^\]]*\]\([^)]*\)|`[^`]*`|https?://\S+)")


def link_opportunity_names(text: str) -> str:
    """Markdown with each opportunity named in the reference snapshot linked to its page, for display only.

    Names match whole words, ignoring case, longest first; text already in a link or code is left alone.
    """
    linked = {r["name"].strip().lower(): r["url"] for r in reference_records()
              if r["name"].strip() and str(r.get("url") or "").startswith("https://")}
    if not text or not linked:
        return text
    names = sorted(linked, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?!\w)", re.IGNORECASE)

    def link(match: re.Match) -> str:
        return f"[{match.group(1)}]({linked[match.group(1).lower()]})"

    parts = LINK_PROTECTED.split(text)  # odd positions are the protected pieces
    return "".join(part if i % 2 else pattern.sub(link, part) for i, part in enumerate(parts))


def receipt_links(result: dict) -> list[str]:
    """Markdown links a connector's receipt gives for one item: the record itself and the record it's on."""
    links = []
    for key, label in (("url", f"Open {result.get('object') or 'record'} ↗"), ("related_url", "Open the record it's on ↗")):
        url = str(result.get(key) or "")
        if url.startswith("https://"):
            links.append(f"[{label}]({url})")
    return links


def describe_reference_record(record: dict) -> str:
    parts = [record.get("account") or "", f"stage {record['stage']}" if record.get("stage") else "",
             f"updated {record['last_updated']}" if record.get("last_updated") else ""]
    return ", ".join(p for p in parts if p)


# Words left out when matching an account name to the notes, so "Acme Ltd" matches "acme.eu".
COMPANY_NAME_SUFFIXES = {
    "ltd", "limited", "inc", "llc", "llp", "plc", "gmbh", "bv", "nv", "sa", "sas", "srl", "ag", "ab", "oy", "as",
    "corp", "corporation", "co", "company", "group", "holdings",
}
NOTE_WORDS = re.compile(r"[a-z0-9]+")


def reference_name_keys(name: str) -> set[str]:
    """A name with spaces and punctuation removed, with and without Ltd/Inc…: "Acme Ltd" -> acmeltd, acme."""
    words = NOTE_WORDS.findall((name or "").lower())
    keys = {"".join(words), "".join(w for w in words if w not in COMPANY_NAME_SUFFIXES)}
    return {key for key in keys if len(key) >= 3}


def notes_reference_records(results: list) -> list[dict]:
    """The snapshot records the notes mention, for the model to name in related_to.

    A record is mentioned when its Opportunity or account name appears in a note's text, file name or folder
    path, ignoring case, spaces, punctuation and Ltd/Inc…, so `acme.eu` and `Ac-Me` both match Acme Ltd.
    Names are compared with runs of up to three words of the notes, so `acme` alone doesn't match "Acme Rockets".
    """
    records = reference_records()
    if not records or not results:
        return []
    runs: set[str] = set()
    for result in results:
        for text in (result.get("text", ""), Path(str(result.get("file_source", ""))).name,
                     result.get("_folder_path", "")):
            words = NOTE_WORDS.findall(str(text or "").lower())
            for size in (1, 2, 3):
                runs.update("".join(words[i:i + size]) for i in range(len(words) - size + 1))
            # Longer names (more than three words) are matched as a whole phrase.
            runs.add(" " + " ".join(words) + " ")
    phrases = [run for run in runs if run.startswith(" ")]

    def mentioned(name: str) -> bool:
        if reference_name_keys(name) & runs:
            return True
        words = NOTE_WORDS.findall((name or "").lower())
        return len(words) > 3 and any(f" {' '.join(words)} " in phrase for phrase in phrases)

    return [r for r in records if mentioned(r["name"]) or mentioned(r.get("account") or "")]


def reference_prompt_text(records: list[dict], next_steps: bool = False) -> str:
    """The open opportunities the notes mention, for the model to name in related_to (empty if there are none).

    With `next_steps`, each line also gives the opportunity's current Next Step, for Agents that update it.
    """
    records = records[:REFERENCE_PROMPT_LIMIT]
    if not records:
        return ""
    lines = "\n".join(
        f"- {r['name']} ({describe_reference_record(r)})"
        + (f" — current Next Step: “{r.get('next_step') or '(empty)'}”" if next_steps else "")
        for r in records
    )
    return (
        "These open opportunities in the connected system are mentioned in the notes. Use this list only to "
        "name the record: when an item is about one of them, set related_to.name to its exact name and "
        "related_to.kind to \"opportunity\". Don't create an item for an opportunity just because it is listed "
        "here; only for what the notes say. If none fits, leave related_to empty rather than inventing a "
        f"name.\n{lines}"
    )


SALESFORCE_CONTEXT_SETTING = "salesforce_context"  # the Ask box's Salesforce chip, kept between sessions
SALESFORCE_PROMPT_HELP = (
    "Whenever this prompt is used, send the Salesforce records for the opportunities the notes mention (stage, "
    "close date, Next Step, ID and link) and your open tasks on them, from the connector's snapshot. Useful for "
    "drafting updates that need the link or close date."
)
SALESFORCE_CONTEXT_LIMIT = 20  # opportunities sent as context, so a small local model isn't swamped


def salesforce_context_records(results: list) -> list[dict]:
    """The snapshot's opportunities that these notes mention, as sent with the Salesforce chip."""
    return notes_reference_records(results)[:SALESFORCE_CONTEXT_LIMIT]


def snapshot_taken(snapshot: dict | None) -> str:
    """When the connector took the snapshot, e.g. "27 Oct 08:00", or ""."""
    try:
        return datetime.datetime.fromisoformat((snapshot or {}).get("generated_at", "")).strftime("%d %b %H:%M")
    except ValueError:
        return ""


def salesforce_context_text(results: list) -> str:
    """Salesforce facts for the opportunities the notes mention, plus my open tasks on them (empty if none).

    Unlike the Agents' list, which only names records, this gives what a question might need: stage, close
    date, Next Step, the record's ID and its link, so the model can quote them.
    """
    records = salesforce_context_records(results)
    if not records:
        return ""
    lines = []
    for r in records:
        facts = [f"Account: {r['account']}" if r.get("account") else "",
                 f"Stage: {r['stage']}" if r.get("stage") else "",
                 f"Close date: {r['close_date']}" if r.get("close_date") else "",
                 f"Next Step: “{r['next_step']}”" if r.get("next_step") else "Next Step: (empty)",
                 f"Last activity: {r['last_activity']}" if r.get("last_activity") else "",
                 f"Last updated: {r['last_updated']}" if r.get("last_updated") else "",
                 f"Salesforce ID: {r['id']}" if r.get("id") else ""]
        lines.append(f"- Opportunity: {r['name']}\n  " + "; ".join(f for f in facts if f)
                     + (f"\n  Link: {r['url']}" if str(r.get("url") or "").startswith("https://") else ""))
    names = {str(r.get(k) or "").strip().lower() for r in records for k in ("name", "account")} - {""}
    tasks = [t for t in reference_open_tasks() if str(t.get("related") or "").strip().lower() in names]
    for t in tasks[:OPEN_TASKS_PROMPT_LIMIT]:
        facts = [f"due {t['due_date']}" if t.get("due_date") else "no due date", t.get("status") or "",
                 f"on {t['related']}" if t.get("related") else ""]
        lines.append(f"- My open task: {t['subject']} (" + ", ".join(f for f in facts if f) + ")"
                     + (f"\n  Link: {t['url']}" if str(t.get("url") or "").startswith("https://") else ""))
    taken = snapshot_taken(load_reference_snapshot())
    return (
        f"Salesforce records for the opportunities these notes mention (a read-only snapshot"
        f"{' taken ' + taken if taken else ''}; the notes may be newer). Use them for facts such as stage, "
        "close date, Next Step and links. When you give a link, copy it exactly from this list and put it on its "
        "own line; never make a link up.\n" + "\n".join(lines)
    )


def with_salesforce_context(search_state: dict, corpus_prompt: str, results: list) -> str:
    """The preamble plus Salesforce records when the search has the Salesforce chip on."""
    if not search_state.get("salesforce_context"):
        return corpus_prompt
    return "\n\n".join(part for part in (corpus_prompt, salesforce_context_text(results)) if part.strip())


def toggle_ask_salesforce_context() -> None:
    save_app_settings({SALESFORCE_CONTEXT_SETTING: not load_app_settings().get(SALESFORCE_CONTEXT_SETTING, False)})


def toggle_follow_up_salesforce_context(search_state: dict) -> None:
    search_state["salesforce_context"] = not search_state.get("salesforce_context")


def render_follow_up_context(search_state: dict, key_prefix: str) -> None:
    """Discussion: extra context for the next follow-ups — the Salesforce records the notes in use mention."""
    snapshot = load_reference_snapshot()
    records = salesforce_context_records(discussion_results(search_state)) if snapshot else []
    on = bool(search_state.get("salesforce_context"))
    taken = snapshot_taken(snapshot)
    count = f"{len(records)} opportunit{'y' if len(records) == 1 else 'ies'}"
    if not snapshot:
        note = "No Salesforce snapshot yet: the connector writes one when it refreshes."
    elif not records:
        note = "The notes in use mention no opportunities in the snapshot."
    elif on:
        note = f"Sending {count} and your open tasks on them · snapshot {taken}"
    else:
        note = f"{count} in the notes in use · snapshot {taken}"
    with st.container(horizontal=True, vertical_alignment="center", gap="xsmall", key=f"{key_prefix}_add_context"):
        st.caption("📎 Add context:", width="content")
        st.button(
            "☁️ Salesforce ✓" if on else "☁️ Salesforce",
            key=f"{key_prefix}_salesforce_context",
            type="primary" if on else "secondary",
            disabled=not snapshot or (not records and not on),
            on_click=toggle_follow_up_salesforce_context,
            args=(search_state,),
            help="Send the opportunities the notes in use mention (stage, close date, Next Step, ID and link) and "
            "your open tasks on them with the next follow-ups. Click again to stop sending them.",
        )
        st.caption(note)


def agent_action(prompt: dict) -> str:
    return f"\"{agent_command(prompt)}\" ({OUTPUT_TYPES[get_prompt_output_type(prompt)].lower()})"


def build_agent_task_prompt(prompts: list[dict], results: list) -> str:
    """Task for an Agent's first answer: its collection prompt, plus how to list items and where it's heading.

    Several Agents (e.g. SF_Next_Step and SF_Tasks) become parts of one request, each answered under its own
    heading, with the items numbered across all parts.
    """
    mentioned = notes_reference_records(results)
    reference = reference_prompt_text(mentioned, any(get_prompt_output_type(p) == "next_step" for p in prompts))
    open_tasks = open_tasks_prompt_text(prompts, mentioned)
    tail = (
        " Ask me the questions you need answered to do that well, and don't guess anything the notes don't say."
        + (f" Say which opportunity each item belongs to.\n\n{reference}" if reference else "")
        + (f"\n\n{open_tasks}" if open_tasks else "")
    )
    if len(prompts) == 1:
        prompt = prompts[0]
        return (
            f"{prompt['text'].strip()}\n\n"
            "Number every item you list (1, 2, 3…) so I can pick them by number, e.g. \"Yes:1,3 ignore 2\". "
            f"We will finish with this action: {agent_action(prompt)}." + tail
        )
    parts = "\n\n".join(
        f"Part {number}: {prompt['name']} (we will finish with {agent_action(prompt)})\n{prompt['text'].strip()}"
        for number, prompt in enumerate(prompts, 1)
    )
    return (
        f"This request has {len(prompts)} parts. Answer each under its own heading (the part's name), and keep "
        "each part to what it asks for.\n\n"
        f"{parts}\n\n"
        "Number every item you list across all parts (1, 2, 3…, continuing from one part to the next) so I can "
        "pick them by number, e.g. \"Yes:1,3 ignore 2\"." + tail
    )


# --- Links (Task executor settings on the Configure page): attached to prompts, shown under the question ------

BEST_GUESS = "{best_guess}"  # in a link's address: replaced by the app's best guess at what you're looking for
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
GUESS_STOPWORDS = {
    "I", "A", "An", "The", "What", "Who", "Whom", "When", "Where", "Why", "How", "Which", "Can", "Could", "Would",
    "Should", "Do", "Does", "Did", "Is", "Are", "Was", "Were", "Please", "List", "Show", "Give", "Tell", "Find",
}
CAPITALISED_WORDS = re.compile(r"\b[A-Z][\w&'.-]*(?:\s+[A-Z][\w&'.-]*)*")


@st.cache_data(ttl=15, show_spinner=False)
def local_service_up(url: str) -> bool:
    """Whether something answers at a localhost address (checked at most every 15 seconds)."""
    import requests

    try:
        requests.get(url, timeout=0.5)
        return True
    except requests.RequestException:
        return False


def link_status_dot(url: str) -> str:
    """🟢 / 🔴 for links to this computer, nothing for anything else."""
    if urlparse(url).hostname not in LOCAL_HOSTS:
        return ""
    return "🟢 " if local_service_up(url.replace(BEST_GUESS, "")) else "🔴 "


def best_guess(question: str, search_filter: dict | None, search_state: dict | None) -> str:
    """What the user is probably looking for, for best guess search links (no LLM involved).

    In order: the Related to names of the Agent's staged or sent records, a reference opportunity whose name
    or account appears in the question (longest first), the first Must include filter term, then capitalised
    names in the question (skipping the question's first word when there's another).
    """
    agent = (search_state or {}).get("agent") or {}
    for batch in [agent.get("staged"), *reversed(agent.get("sent") or [])]:
        for item in (batch or {}).get("items", []):
            name = str((item.get("related_to") or {}).get("name", "")).strip()
            if name:
                return name
    asked = (question or "").lower()
    if asked:
        mentioned = [
            (len(term), record["name"]) for record in reference_records()
            for term in (record["name"], record.get("account") or "") if len(term) >= 3 and term.lower() in asked
        ]
        if mentioned:
            return max(mentioned)[1]
    must_include = (search_filter or {}).get("all") or []
    if must_include:
        return str(must_include[0])
    candidates = []
    for match in CAPITALISED_WORDS.finditer(question or ""):
        words = [w.strip(".,;:!?") for w in match.group().split() if w.strip(".,;:!?") not in GUESS_STOPWORDS]
        if words:
            starts_sentence = match.start() == 0 or (question or "")[: match.start()].rstrip()[-1:] in ".?!"
            candidates.append((starts_sentence, " ".join(words)))
    later = [text for starts_sentence, text in candidates if not starts_sentence]
    return (later or [text for _, text in candidates] or [""])[0]


def render_page_title(title: str) -> None:
    """Title for every page except the main one; its 🧠 goes back to the main screen in the same tab."""
    st.markdown(
        f'<h1><a href="./" target="_self" title="Back to RAG Workspace Manager" '
        f'style="text-decoration:none">🧠</a> {title}</h1>',
        unsafe_allow_html=True,
    )


# The task executor is the one special link: the service that processes the outbox (e.g. the Salesforce
# Connector app). It's defined first in Task executor settings, listed first everywhere, and on the Actions page.
EXECUTOR_ROLE = "executor"


def ordered_links(library: dict) -> list[dict]:
    """The links (Task executor settings) with the task executor first."""
    return sorted(library.get("links", []), key=lambda link: link.get("role") != EXECUTOR_ROLE)


def executor_link(library: dict) -> dict | None:
    return next((link for link in library.get("links", []) if link.get("role") == EXECUTOR_ROLE), None)


def link_markdown(link: dict, guess: str = "") -> str:
    """One link as Markdown (opens in a new tab), with its status dot and any best guess filled in."""
    title, url = link.get("title") or link["url"], link["url"]
    if BEST_GUESS in url:
        url = url.replace(BEST_GUESS, quote_plus(guess))
        title = f"{title}: {guess}" if guess else title
    title = title.replace("[", "\\[").replace("]", "\\]")
    return f"{link_status_dot(link['url'])}[{title}]({url.replace(' ', '%20').replace(')', '%29')})"


def render_prompt_links(prompts: list[dict], library: dict, guess: str, width="stretch") -> None:
    """The injected prompts' links, inline (task executor first), each opening in a new tab."""
    chosen = {link_id for prompt in prompts for link_id in prompt.get("links", [])}
    if not chosen:
        return
    parts = [link_markdown(link, guess) for link in ordered_links(library) if link["id"] in chosen and link.get("url")]
    if parts:
        st.markdown(" · ".join(parts), width=width)


# --- Reference documents: notes sent to the LLM as background, attached to prompts or added for one question --

REFERENCE_PAGE_URL = "?page=reference"
REFERENCE_DOC_CHAR_LIMIT = 20_000  # characters of one reference document (its notes together) sent to the model
REFERENCE_RECENT = 5
REFERENCE_ROW_CHARS = 46  # about what fits on the Ask box's Reference row; titles that would overflow are left out
EXPORT_STATE_FILE = ".notes-export-state.json"  # the Notes exporter's {Apple Notes ID: file} map
ACTIVE_REFERENCES_KEY = "active_reference_ids"  # documents added in the Ask box for this session's questions


def reference_documents(library: dict | None = None) -> list[dict]:
    return (library or load_prompt_library()).get("references", [])


def find_reference_document(ref_id: str | None, library: dict | None = None) -> dict | None:
    return next((r for r in reference_documents(library) if r["id"] == ref_id), None) if ref_id else None


def reference_label(ref: dict) -> str:
    return ref.get("title") or "Untitled document"


def reference_note_key(note: dict) -> str:
    return note.get("note_id") or note.get("path", "")


def update_reference_documents(change) -> None:
    library = load_prompt_library()
    library["references"] = change([dict(ref) for ref in library.get("references", [])])
    save_prompt_library(library)


def add_reference_document() -> None:
    update_reference_documents(lambda refs: refs + [{"id": uuid.uuid4().hex[:8], "title": "", "notes": []}])


def delete_reference_document(ref_id: str) -> None:
    """Delete a reference document and detach it from every prompt and from the Ask box."""
    library = load_prompt_library()
    library["references"] = [ref for ref in library.get("references", []) if ref["id"] != ref_id]
    for prompt in library["prompts"]:
        prompt["references"] = [i for i in prompt.get("references", []) if i != ref_id]
    save_prompt_library(library)
    remove_active_reference(ref_id)


def save_reference_title(ref_id: str) -> None:
    title = str(st.session_state.get(f"reference_title_{ref_id}", "")).strip()
    update_reference_documents(lambda refs: [{**ref, "title": title} if ref["id"] == ref_id else ref for ref in refs])


def toggle_reference_note(ref_id: str, note: dict) -> None:
    """Add a note to a reference document, or take it out if it's already there."""
    def change(refs: list[dict]) -> list[dict]:
        for ref in refs:
            if ref["id"] == ref_id:
                notes = ref.get("notes", [])
                there = any(reference_note_key(n) == reference_note_key(note) for n in notes)
                ref["notes"] = [n for n in notes if reference_note_key(n) != reference_note_key(note)] if there \
                    else notes + [note]
        return refs
    update_reference_documents(change)


def remove_reference_note(ref_id: str, key: str) -> None:
    update_reference_documents(lambda refs: [
        {**ref, "notes": [n for n in ref.get("notes", []) if reference_note_key(n) != key]} if ref["id"] == ref_id
        else ref for ref in refs
    ])


@st.cache_data(ttl=60, show_spinner=False)
def export_state_paths(corpus_root: str) -> tuple[str, dict[str, str]]:
    """(the Notes exporter's output folder, {Apple Notes ID: file path in it}) from its state file in the corpus
    root or up to three folders above; empty when there's none (notes not exported with the exporter)."""
    if not corpus_root:
        return "", {}
    folder = Path(corpus_root).expanduser().resolve()
    for candidate in [folder, *list(folder.parents)[:3]]:
        state = candidate / EXPORT_STATE_FILE
        if state.is_file():
            try:
                notes = json.loads(state.read_text()).get("notes") or {}
            except (OSError, ValueError):
                return "", {}
            return str(candidate), {nid: e["path"] for nid, e in notes.items() if isinstance(e, dict) and e.get("path")}
    return "", {}


def reference_note_path(note: dict) -> Path | None:
    """Where a reference note is now: found by its Apple Notes ID (so renames and moves keep it), else its path."""
    folder, paths = export_state_paths(str(get_corpus_root_path() or ""))
    if note.get("note_id") in paths:
        path = Path(folder) / paths[note["note_id"]]
        if path.is_file():
            return path
    path = Path(note.get("path") or "")
    return path if note.get("path") and path.is_file() else None


def reference_note_for_path(path: Path) -> dict:
    """A reference note record for a note file: its Apple Notes ID (when the exporter has it), title and path."""
    folder, paths = export_state_paths(str(get_corpus_root_path() or ""))
    note_id = ""
    if folder:
        try:
            rel = path.resolve().relative_to(Path(folder)).as_posix()
        except ValueError:
            rel = ""
        note_id = next((nid for nid, file in paths.items() if file == rel), "")
    return {"note_id": note_id, "title": path.stem, "path": str(path)}


def reference_note_folder(path: Path) -> str:
    """'Clients > Prospects > Acme' for a note file, relative to the corpus root."""
    root = get_corpus_root_path()
    try:
        parts = path.parent.resolve().relative_to(Path(root).resolve()).parts if root else ()
    except ValueError:
        parts = (path.parent.name,)
    return FOLDER_PATH_SEPARATOR.join(parts)


def reference_document_text(ref: dict) -> tuple[str, list[str]]:
    """A reference document's notes as one text for the model (trimmed to the limit), and the notes not found."""
    parts, missing = [], []
    for note in ref.get("notes", []):
        path = reference_note_path(note)
        try:
            parts.append(path.read_text(errors="ignore").strip())
        except (AttributeError, OSError):
            missing.append(note.get("title") or reference_note_key(note))
    text = "\n\n".join(parts)
    if len(text) > REFERENCE_DOC_CHAR_LIMIT:
        text = text[:REFERENCE_DOC_CHAR_LIMIT].rstrip() + "\n[… trimmed]"
    return text, missing


def run_reference_documents(tasks: list[dict], library: dict) -> list[dict]:
    """This run's reference documents: those of the injected prompts, then those added in the Ask box."""
    ids = [i for task in tasks for i in task.get("references", [])] + list(st.session_state.get(ACTIVE_REFERENCES_KEY, []))
    found = (find_reference_document(i, library) for i in dict.fromkeys(ids))
    return [ref for ref in found if ref and ref.get("notes")]


def mark_references_used(ref_ids: list[str]) -> None:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    update_reference_documents(lambda refs: [{**ref, "last_used": now} if ref["id"] in ref_ids else ref for ref in refs])


def recent_reference_documents(library: dict) -> list[dict]:
    """Up to five most recently used documents, leaving out any whose title would push the row past one line."""
    used = sorted((r for r in reference_documents(library) if r.get("last_used")), key=lambda r: r["last_used"],
                  reverse=True)[:REFERENCE_RECENT]
    shown, width = [], 0
    for ref in used:
        cost = len(reference_label(ref)) + 4  # a chip's padding
        if width + cost <= REFERENCE_ROW_CHARS:
            shown.append(ref)
            width += cost
    return shown


def toggle_active_reference(ref_id: str) -> None:
    active = list(st.session_state.get(ACTIVE_REFERENCES_KEY, []))
    st.session_state[ACTIVE_REFERENCES_KEY] = [i for i in active if i != ref_id] if ref_id in active else active + [ref_id]


def remove_active_reference(ref_id: str) -> None:
    st.session_state[ACTIVE_REFERENCES_KEY] = [i for i in st.session_state.get(ACTIVE_REFERENCES_KEY, []) if i != ref_id]


def pick_reference_from_finder() -> None:
    ref_id = st.session_state.get("reference_finder")
    if ref_id and ref_id not in st.session_state.get(ACTIVE_REFERENCES_KEY, []):
        toggle_active_reference(ref_id)
    st.session_state["reference_finder"] = None


def reference_view_link(ref: dict) -> str:
    """An HTML link opening a reference document in its own tab."""
    return (f'<a href="{REFERENCE_PAGE_URL}&ref={ref["id"]}" target="_blank" rel="noopener">'
            f"📄 {html.escape(reference_label(ref))}</a>")


def render_ask_reference_row(library: dict, tasks: list[dict]) -> None:
    """Ask box: reference documents for this question — recent ones as chips, a finder for the rest, and what
    will be sent (with those the injected prompts bring)."""
    refs = reference_documents(library)
    snapshot = load_reference_snapshot()
    if not refs and not snapshot:
        return
    active = list(st.session_state.get(ACTIVE_REFERENCES_KEY, []))
    from_prompts = {i: task["name"] for task in tasks for i in task.get("references", [])}
    salesforce_on = bool(load_app_settings().get(SALESFORCE_CONTEXT_SETTING))
    salesforce_prompts = [task["name"] for task in tasks if task.get("salesforce")]
    with st.container(horizontal=True, vertical_alignment="center", gap="xsmall", key="ask_reference_row"):
        st.caption("Reference:")
        if snapshot:
            st.button(
                "☁️ Salesforce ✓" if salesforce_on or salesforce_prompts else "☁️ Salesforce",
                key="use_salesforce_context",
                type="primary" if salesforce_on or salesforce_prompts else "secondary",
                on_click=toggle_ask_salesforce_context,
                disabled=bool(salesforce_prompts) and not salesforce_on,
                help=(f"Sent with {', '.join(salesforce_prompts)}." if salesforce_prompts and not salesforce_on else
                      "Send the Salesforce records for the opportunities the found notes mention (stage, close "
                      "date, Next Step, ID and link) and your open tasks on them, from the connector's snapshot "
                      f"taken {snapshot_taken(snapshot) or 'earlier'}. Stays on until you click it again."),
            )
        for ref in recent_reference_documents(library):
            st.button(
                reference_label(ref) + (" ✓" if ref["id"] in active or ref["id"] in from_prompts else ""),
                key=f"use_reference_{ref['id']}",
                type="primary" if ref["id"] in active or ref["id"] in from_prompts else "secondary",
                on_click=toggle_active_reference,
                args=(ref["id"],),
                disabled=ref["id"] in from_prompts and ref["id"] not in active,
                help=f"Sent with {from_prompts[ref['id']]}." if ref["id"] in from_prompts
                else "Send this reference document with your question.",
            )
        labels = {ref["id"]: reference_label(ref) for ref in refs}
        if refs:
            st.selectbox(
                "Find a reference document",
                options=list(labels),
                index=None,
                format_func=labels.get,
                key="reference_finder",
                on_change=pick_reference_from_finder,
                placeholder="Find a reference document…",
                label_visibility="collapsed",
            )
    sending = run_reference_documents(tasks, library)
    salesforce_sending = bool(snapshot) and (salesforce_on or bool(salesforce_prompts))
    if sending or salesforce_sending:
        st.markdown(
            "Sending: " + " · ".join(
                (["☁️ Salesforce records"
                  + (f" <small>({html.escape(', '.join(salesforce_prompts))})</small>"
                     if salesforce_prompts and not salesforce_on else "")] if salesforce_sending else [])
                + [reference_view_link(ref) + (f" <small>({html.escape(from_prompts[ref['id']])})</small>"
                                               if ref["id"] in from_prompts else "")
                   for ref in sending]
            ),
            unsafe_allow_html=True,
        )
        for ref in sending:
            if ref["id"] in active and ref["id"] not in {r["id"] for r in recent_reference_documents(library)}:
                st.button(f"✕ {reference_label(ref)}", key=f"drop_reference_{ref['id']}", type="tertiary",
                          on_click=remove_active_reference, args=(ref["id"],), help="Don't send this one")


def render_reference_documents_section(library: dict) -> None:
    """Prompt library: the reference documents, each a short title and its notes (folded)."""
    refs = library.get("references", [])
    with st.expander(
        f"Reference documents · {len(refs)}",
        expanded=st.query_params.get("section") == "references",
        key="prompt_page_references_open",
        icon=":material/description:",
    ):
        st.caption(
            "Notes the LLM gets as background, e.g. a process overview. Give each document a short title, pick its "
            "notes in the search page (**Choose notes**, new tab) and check them with **Open**. Attach documents to "
            "prompts with **Reference documents** on each card, and they're sent whenever that prompt is used; or "
            "add one for a question in the Ask box. A note is kept by its Apple Notes ID (the code shown), so "
            f"renaming or moving it keeps the link. Up to {REFERENCE_DOC_CHAR_LIMIT:,} characters of each document "
            "are sent."
        )
        for ref in refs:
            with st.container(border=True, key=f"reference_card_{ref['id']}"):
                with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
                    st.text_input("Short title", value=ref.get("title", ""), key=f"reference_title_{ref['id']}",
                                  on_change=save_reference_title, args=(ref["id"],), max_chars=40,
                                  placeholder="e.g. PoC types")
                    st.button("Choose notes", key=f"reference_choose_{ref['id']}", icon=":material/search:",
                              on_click=go_to_pick, args=(ref["id"],),
                              help="Goes to the search page to add or remove this document's notes; **Done** "
                              "brings you back here.")
                    if st.button("Open", key=f"reference_open_{ref['id']}", icon=":material/visibility:",
                                 disabled=not ref.get("notes"), help="Read its notes, as the LLM gets them."):
                        reference_popup(ref["id"])
                    st.button(":material/delete:", key=f"reference_delete_{ref['id']}", type="tertiary",
                              on_click=delete_reference_document, args=(ref["id"],),
                              help="Delete this reference document and detach it from prompts.")
                if not ref.get("notes"):
                    st.caption("No notes yet: **Choose notes** opens the search page to pick them.")
                for note in ref.get("notes", []):
                    path = reference_note_path(note)
                    code = f" · `{note['note_id'][:8]}`" if note.get("note_id") else " · (no Apple Notes ID: kept by path)"
                    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                        if path:
                            st.markdown(f"📄 {md_escape_text(path.stem)} · {reference_note_folder(path)}{code}",
                                        width="stretch")
                        else:
                            st.markdown(f"⚠ *{md_escape_text(note.get('title', 'a note'))}* wasn't found: deleted, or "
                                        "no longer exported?", width="stretch")
                        st.button(":material/close:", key=f"reference_note_drop_{ref['id']}_{reference_note_key(note)}",
                                  type="tertiary", on_click=remove_reference_note,
                                  args=(ref["id"], reference_note_key(note)), help="Take this note out")
                if ref.get("notes"):
                    text, _missing = reference_document_text(ref)
                    st.caption(f"{len(text):,} characters sent" + (
                        f" (trimmed to {REFERENCE_DOC_CHAR_LIMIT:,})" if text.endswith("[… trimmed]") else ""))
        st.button("Add reference document", icon=":material/add:", key="reference_add", on_click=add_reference_document)


def md_escape_text(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#|~])", r"\\\1", text)


def render_reference_page() -> None:
    """?page=reference&ref=<id>: a reference document's notes in full, opened in its own tab."""
    library = load_prompt_library()
    ref = find_reference_document(st.query_params.get("ref"), library)
    render_page_title("Reference document")
    if ref is None:
        st.warning("This reference document doesn't exist (any more).")
        return
    render_reference_document_body(ref)


DOC_PARAM = "doc"  # ?doc=<Apple Notes ID or path>: the note open in the pick mode popup (refresh reopens it)


def open_document_popup(note: dict) -> None:
    st.query_params[DOC_PARAM] = note.get("note_id") or note.get("path", "")


def close_document_popup() -> None:
    st.query_params.pop(DOC_PARAM, None)
    for key in [k for k in st.session_state if str(k).startswith("doc_pending_")]:
        del st.session_state[key]


def note_for_doc_param(value: str) -> tuple[dict, Path] | None:
    """The note a ?doc= value names: an Apple Notes ID, or the path of a note under the corpus root."""
    if not value:
        return None
    path = reference_note_path({"note_id": value})
    root = get_corpus_root_path()
    if path is None and root:
        candidate = Path(value)
        if candidate.suffix == ".md" and candidate.is_file() and candidate.resolve().is_relative_to(Path(root).resolve()):
            path = candidate
    return (reference_note_for_path(path), path) if path else None


def render_note_body(path: Path, note: dict, height: int = 440) -> None:
    modified = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%a %d %b %Y %H:%M")
    st.caption(f"{reference_note_folder(path)} · modified {modified}"
               + (f" · Apple Notes ID `{note['note_id']}`" if note.get("note_id") else ""))
    with st.container(height=height, border=True):
        st.markdown(path.read_text(errors="ignore"))


@st.dialog("Note", width="large", on_dismiss=close_document_popup)
def document_popup(ref_id: str, doc: str) -> None:
    """Pick mode: a note in a popup, with Add to <document> → Apply and return / Remove from list."""
    ref, found = find_reference_document(ref_id), note_for_doc_param(doc)
    if ref is None or found is None:
        st.warning("That note or reference document can't be found (any more).")
        if st.button("Close"):
            close_document_popup()
            st.rerun()
        return
    note, path = found
    label = reference_label(ref)
    st.markdown(f"**{md_escape_text(path.stem)}**")
    render_note_body(path, note)
    key = reference_note_key(note)
    pending_key = f"doc_pending_{key}"
    in_list = any(reference_note_key(n) == key for n in ref.get("notes", []))
    # Add and Remove change state in callbacks, so the popup redraws with the right buttons; Apply and return
    # and Return rerun the page, which closes it.
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.session_state.get(pending_key):  # added here, not saved until Apply and return
            if st.button("Apply and return", key="doc_apply", type="primary", icon=":material/check:",
                         on_click=apply_document_popup, args=(ref_id, note, pending_key)):
                st.rerun()
            st.button("Remove from list", key="doc_unadd", icon=":material/remove:",
                      on_click=st.session_state.pop, args=(pending_key, None))
        elif in_list:
            st.caption(f"✓ In {label}")
            st.button("Remove from list", key="doc_remove", icon=":material/remove:",
                      on_click=toggle_reference_note, args=(ref_id, note))
            if st.button("Return", key="doc_return", on_click=close_document_popup):
                st.rerun()
        else:
            st.button(f"Add to {label}", key="doc_add", type="primary", icon=":material/add:",
                      on_click=st.session_state.__setitem__, args=(pending_key, True))


def apply_document_popup(ref_id: str, note: dict, pending_key: str) -> None:
    """Apply and return: save the note into the reference document, then close the popup."""
    ref = find_reference_document(ref_id)
    if ref and not any(reference_note_key(n) == reference_note_key(note) for n in ref.get("notes", [])):
        toggle_reference_note(ref_id, note)
    close_document_popup()


@st.dialog("Reference document", width="large")
def reference_popup(ref_id: str) -> None:
    """A reference document's notes in a popup (from the prompt library or the pick panel)."""
    ref = find_reference_document(ref_id)
    if ref is None:
        st.warning("This reference document doesn't exist (any more).")
        return
    render_reference_document_body(ref, height=320)


def render_reference_document_body(ref: dict, height: int | None = None) -> None:
    """A reference document: its title, size as sent and each note in full."""
    text, _missing = reference_document_text(ref)
    st.subheader(reference_label(ref))
    st.caption(f"{len(ref.get('notes', []))} note(s) · {len(text):,} characters sent to the LLM"
               + (f", trimmed to {REFERENCE_DOC_CHAR_LIMIT:,}" if text.endswith("[… trimmed]") else ""))
    for note in ref.get("notes", []):
        path = reference_note_path(note)
        if path is None:
            st.warning(f"Not found: *{md_escape_text(note.get('title', 'a note'))}* — deleted, or no longer exported?")
            continue
        st.markdown(f"**{md_escape_text(path.stem)}**")
        if height:
            render_note_body(path, note, height)
        else:
            modified = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%a %d %b %Y %H:%M")
            st.caption(f"{reference_note_folder(path)} · modified {modified}"
                       + (f" · Apple Notes ID `{note['note_id']}`" if note.get("note_id") else ""))
            st.markdown(path.read_text(errors="ignore"))


def go_to_pick(ref_id: str) -> None:
    """Choose notes: the search page in pick mode, in this tab."""
    st.query_params.from_dict({"pick": ref_id})


def return_from_pick() -> None:
    """Done: back to the prompt library's Reference documents, showing the document's notes."""
    close_document_popup()
    st.session_state["prompt_page_references_open"] = True
    st.query_params.from_dict({"page": "prompts", "section": "references"})


def render_pick_done(ref: dict, key: str) -> None:
    st.button("✓ Done — back to Reference documents", key=key, type="primary", icon=":material/arrow_back:",
              on_click=return_from_pick, use_container_width=True,
              help=f"Goes back to the prompt library, where **{reference_label(ref)}** shows the notes you chose.")


def render_pick_panel(ref: dict) -> None:
    """Search page in pick mode (?pick=<id>): the reference document being filled, under the Filter box."""
    ref = find_reference_document(ref["id"]) or ref  # re-read: notes are added from the list below
    with st.container(border=True, key="pick_panel"):
        st.markdown(f"**Choosing notes for:** {md_escape_text(reference_label(ref))}")
        st.caption(
            "Find notes with the Filter box above (**Find notes** or **Filter search only**), then **Add** them in "
            "the list below. Changes are saved straight away; **Done** takes you back to Reference documents."
        )
        for note in ref.get("notes", []):
            path = reference_note_path(note)
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"📄 {md_escape_text(path.stem if path else note.get('title', 'a note'))}"
                            + ("" if path else " — ⚠ not found"), width="stretch")
                st.button(":material/close:", key=f"pick_drop_{reference_note_key(note)}", type="tertiary",
                          on_click=remove_reference_note, args=(ref["id"], reference_note_key(note)))
        if not ref.get("notes"):
            st.caption("No notes yet.")
        with st.container(horizontal=True, vertical_alignment="center"):
            if ref.get("notes") and st.button("Open", key="pick_panel_open", icon=":material/visibility:"):
                reference_popup(ref["id"])
            render_pick_done(ref, "pick_done_top")


def save_executor_field(field: str) -> None:
    """Save the task executor's title or address, creating the executor link the first time."""
    library = load_prompt_library()
    link = executor_link(library)
    if link is None:
        link = {"id": uuid.uuid4().hex[:8], "title": "", "url": "", "role": EXECUTOR_ROLE}
        library["links"].insert(0, link)
    link[field] = str(st.session_state.get(f"executor_{field}", "")).strip()
    save_prompt_library(library)


def update_links(change) -> None:
    library = load_prompt_library()
    library["links"] = change(list(library.get("links", [])))
    save_prompt_library(library)


def add_link() -> None:
    update_links(lambda links: links + [{"id": uuid.uuid4().hex[:8], "title": "", "url": ""}])


def delete_link(link_id: str) -> None:
    library = load_prompt_library()
    library["links"] = [link for link in library.get("links", []) if link["id"] != link_id]
    for prompt in library["prompts"]:
        prompt["links"] = [i for i in prompt.get("links", []) if i != link_id]
    save_prompt_library(library)


def save_link_field(link_id: str, field: str) -> None:
    value = str(st.session_state.get(f"link_{field}_{link_id}", "")).strip()
    update_links(lambda links: [{**link, field: value} if link["id"] == link_id else link for link in links])


TITLE_MARKER_POSITIONS = {"prefix": "Before the title", "suffix": "After the title"}


def get_title_marker(library: dict) -> dict:
    """The prompt library's marker for Agent-created titles: {"text": "", "position": "prefix"|"suffix"}."""
    marker = library.get("title_marker") or {}
    position = marker.get("position") if marker.get("position") in TITLE_MARKER_POSITIONS else "prefix"
    return {"text": str(marker.get("text", "")).strip(), "position": position}


def marked_title(title: str, marker: dict) -> str:
    text, title = marker["text"], title.strip()
    if not text or not title or text.lower() in title.lower():  # none set, nothing to mark, or already there
        return title
    return f"{text} {title}" if marker["position"] == "prefix" else f"{title} {text}"


def apply_title_marker(item: dict, marker: dict) -> dict:
    """Add the marker to a staged task's subject, note's title or email's subject."""
    field = "title" if item.get("type") == "note" else "subject"
    return {**item, field: marked_title(item.get(field, ""), marker)} if field in item else item


def profile_short_name(library: dict) -> str:
    """Your short name (e.g. initials), set under About me; it signs Next Step updates."""
    return str((library.get("profile") or {}).get("short_name", "")).strip()


def next_step_text(text: str, short_name: str, today: datetime.date, marker: dict) -> str:
    """'2026-10-25-JDOE: waiting on client…', plus the title marker."""
    signed = f"{today:%Y-%m-%d}-{short_name}" if short_name else f"{today:%Y-%m-%d}"
    return marked_title(f"{signed}: {text.strip()}", marker)


def expand_next_steps(entries: list[dict], marker: dict, short_name: str, today: datetime.date) -> list[dict]:
    """Each model entry becomes a Next Step update and a task to chase it, both on the same opportunity.

    The update carries `previous` (the Next Step in the reference snapshot) so the connector only overwrites
    what the user saw; without a snapshot entry it's left out.
    """
    items = []
    for entry in entries:
        text = str(entry.get("next_step", "") or "").strip()
        if not text:
            continue
        related = normalise_related_to(entry.get("related_to"))
        if related["name"] and not related["kind"]:
            related["kind"] = "opportunity"  # next steps live on opportunities
        update = {
            "type": "update",
            "field": "next_step",
            "value": next_step_text(text, short_name, today, marker),
            "related_to": related,
        }
        record = find_reference_record(related["name"]) if related["kind"] == "opportunity" else None
        if record is not None:
            update["previous"] = record.get("next_step", "")
            if entry.get("already_said") is True:
                update["_already_said"] = True  # the model's view; staging only, never sent
        chase = normalise_draft_item("task", {
            "subject": str(entry.get("chase_subject", "") or "").strip() or f"Chase: {text}",
            "description": str(entry.get("chase_description", "") or "").strip() or text,
            "due_date": entry.get("chase_due_date", ""),
            "related_to": related,
        })
        items += [update, apply_title_marker(chase, marker)]
    return items


def stage_agent_batch(
    search_state: dict,
    prompts: list[dict],
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str,
    last_note: str = "",
) -> dict:
    """Stage every injected Agent's records into one batch (one package). Each Agent is asked separately."""
    parts = len(prompts)
    batches = [
        stage_agent_items(search_state, prompt, model_name, max_tokens, temperature, corpus_prompt,
                          part=(number, parts) if parts > 1 else None, last_note=last_note)
        for number, prompt in enumerate(prompts, 1)
    ]
    if parts == 1:
        return batches[0]
    items = [item for batch in batches for item in batch["items"]]
    types = {item["type"] for item in items}
    return {
        "id": uuid.uuid4().hex,
        "type": (types.pop() if len(types) == 1 else "mixed") if items else batches[0]["type"],
        "summary": " ".join(batch["summary"] for batch in batches if batch["summary"]),
        "items": items,
        "model": model_name,
    }


def stage_agent_items(
    search_state: dict,
    prompt: dict,
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str,
    part: tuple[int, int] | None = None,
    last_note: str = "",
) -> dict:
    """Turn the discussion so far into exact records for the Agent's Action Command. Nothing is sent.

    `part` is (number, of) when several Agents were injected: only this Agent's part is staged here.
    `last_note` is the One last note typed above Execute, e.g. "exclude points 2-5": it overrides the discussion.
    """
    item_type = get_prompt_output_type(prompt)
    messages = build_discussion_messages(search_state, max_tokens, temperature, corpus_prompt)
    mentioned = notes_reference_records(discussion_results(search_state))
    reference = reference_prompt_text(mentioned, item_type == "next_step")
    open_tasks = open_tasks_prompt_text([prompt], mentioned)
    only_part = (
        f"This covers only part {part[0]} of {part[1]} ({prompt['name']}): give only that part's records; "
        "the other parts are staged separately.\n\n" if part else ""
    )
    messages.append({
        "role": "user",
        "content": (
            f"{agent_command(prompt)}\n\n{only_part}Turn what we have agreed in this conversation into records now. "
            "Include only the items I selected or agreed, with the details we agreed, and leave out anything "
            "I said to ignore. "
            + (f"One last note, which overrides anything above where they differ: {last_note.strip()} "
               if last_note.strip() else "")
            + f"{STRUCTURED_OUTPUT_GUIDES[item_type]} {STRUCTURED_RELATED_GUIDE}"
            + (f"\n\n{reference}" if reference else "")
            + (f"\n\n{open_tasks}" if open_tasks else "")
        ),
    })
    reply, _usage = chat_with_ollama(
        messages,
        model_name,
        max_tokens=max(max_tokens, 1024),  # several records in JSON need more room than a short answer
        temperature=temperature,
        num_ctx=FOLLOW_UP_NUM_CTX,
        response_format=structured_output_schema(item_type),
    )
    try:
        data = json.loads(reply)
    except ValueError as exc:
        raise RuntimeError(
            f"`{model_name}` didn't return complete JSON — usually it ran out of tokens. "
            "Raise **Max tokens** and try again."
        ) from exc
    library = load_prompt_library()
    marker = get_title_marker(library)
    entries = [item for item in data.get("items", []) if isinstance(item, dict)]
    if item_type == "next_step":
        items = expand_next_steps(entries, marker, profile_short_name(library), datetime.date.today())
    else:
        items = [apply_title_marker(normalise_draft_item(item_type, item), marker) for item in entries]
    for item in items:
        item["_agent_id"] = prompt["id"]  # which Agent staged it, for the actions log; not sent
    return {
        "id": uuid.uuid4().hex,  # the package id if this is sent; each staging is a new package
        "type": "mixed" if item_type == "next_step" else item_type,
        "summary": str(data.get("summary", "")).strip(),
        "items": items,
        "model": model_name,
    }


def get_outbox_dir() -> Path:
    raw = str(load_app_settings().get("outbox_path", "") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_OUTBOX_DIR


def persist_outbox_path() -> None:
    save_app_settings({"outbox_path": str(st.session_state.get("outbox_path_input", "")).strip()})


def validate_handoff_package(package: dict) -> None:
    """Check a package against its schema (schemas/handoff.v1 or v2) before it is written."""
    import jsonschema

    path = HANDOFF_SCHEMA_PATHS.get(package.get("schema"))
    if path is None:
        raise ValueError(f"unknown package format {package.get('schema')!r}")
    jsonschema.validate(package, json.loads(path.read_text()))


def write_handoff_package(package: dict, outbox: Path) -> Path:
    """Validate, then write atomically so a connector never reads a half-written file."""
    validate_handoff_package(package)
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.fromisoformat(package["created"]).strftime("%Y%m%d-%H%M%S")
    target = outbox / f"{stamp}-{package['item_type']}-{package['id'][:8]}.json"
    temp = outbox / f".{target.name}.tmp"
    temp.write_text(json.dumps(package, indent=2, ensure_ascii=False))
    temp.replace(target)
    return target


def build_handoff_package(
    search_state: dict, prompts: list[dict], staged: dict, source_notes: list[tuple[str, Path | None]]
) -> dict:
    return {
        # v1 unless the package holds updates, mixed items or Create anyway, so v1-only connectors keep working.
        "schema": HANDOFF_SCHEMA_V2_ID if staged["type"] in ("update", "mixed")
        or any(item.get("allow_duplicate") for item in staged["items"]) else HANDOFF_SCHEMA_ID,
        "id": staged["id"],
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "item_type": staged["type"],
        "source": {
            "library": search_state.get("library_display") or search_state["library"],
            "library_key": search_state["library"],
            "question": search_state["question"],
            "prompt": {  # several Agents read e.g. "SF_Next_Step + SF_Tasks"
                "id": "+".join(p["id"] for p in prompts), "name": " + ".join(p["name"] for p in prompts),
            },
            "model": staged["model"],
            "notes": [{"file": file, **({"path": str(path)} if path else {})} for file, path in source_notes],
        },
        "items": [{k: v for k, v in item.items() if not k.startswith("_")} for item in staged["items"]],
    }


def read_handoff_receipt(package_file: Path) -> dict | None:
    """The receipt a connector left beside the package in processed/ or failed/, if there is one yet."""
    for folder in HANDOFF_RESULT_DIRS:
        receipt = package_file.parent / folder / f"{package_file.stem}.receipt.json"
        if receipt.is_file():
            try:
                return json.loads(receipt.read_text())
            except ValueError:
                return None
    return None


def pending_outbox_files(outbox: Path) -> list[Path]:
    """Packages waiting in the outbox (connectors move them to processed/ or failed/ when done)."""
    if not outbox.is_dir():
        return []
    return [p for p in outbox.glob("*.json") if not p.name.startswith(".")]


def latest_receipt_time(folder: Path) -> datetime.datetime | None:
    times = []
    for path in folder.glob("*.receipt.json"):
        try:
            when = datetime.datetime.fromisoformat(json.loads(path.read_text())["processed_at"])
        except (OSError, ValueError, KeyError, TypeError):
            when = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        times.append(when if when.tzinfo else when.astimezone())
    return max(times) if times else None


def format_when(when: datetime.datetime | None) -> str:
    return when.strftime("%d %b %H:%M") if when else "never"


def render_outbox_status(link: bool = True) -> None:
    """One line at the top of every page: what's waiting in the outbox, what needs attention, and when a
    connector last processed it, with a link to the Actions page."""
    outbox = get_outbox_dir()
    waiting = pending_outbox_files(outbox)
    items = 0
    for path in waiting:
        try:
            items += len(json.loads(path.read_text()).get("items", []))
        except (OSError, ValueError, AttributeError):
            pass
    attention = len(list((outbox / "failed").glob("*.receipt.json")))

    st.caption(
        f":material/outbox: **Outbox:** {len(waiting)} waiting ({items} item{'' if items == 1 else 's'})"
        + (f" · :orange[**{attention} need{'s' if attention == 1 else ''} attention**]" if attention else "")
        + f" · Last processed {format_when(latest_receipt_time(outbox / 'processed'))}"
        + f" · Last error {format_when(latest_receipt_time(outbox / 'failed'))}"
        + (f" · [Open Actions]({ACTIONS_PAGE_URL})" if link else ""),
        help=(
            f"Folder: `{outbox}`. A connector (e.g. Salesforce) processes what's waiting. Errors and "
            "\"need attention\" include items held for you to choose a record. **Open Actions** lists everything."
        ),
    )


ACTIONS_PAGE_URL = "?page=actions"
OUTBOX_STATES = {  # tab order on the Actions page: what needs attention first
    "failed": ("Needs attention", "Nothing needs attention."),
    "waiting": ("Waiting", "Nothing is waiting for a connector."),
    "processed": ("Processed", "Nothing has been processed yet."),
    "withdrawn": ("Withdrawn", "Nothing has been withdrawn."),
}
# Withdrawn packages are moved here, out of connectors' reach (they only read the outbox's top level).
WITHDRAWN_DIR = "withdrawn"
PACKAGE_STATUS = {
    "done": ("✅", "done"),
    "partial": ("⚠️", "partly done"),
    "needs_review": ("⚠️", "needs review"),
    "failed": ("❌", "failed"),
    "waiting": ("⏳", "waiting"),
    "withdrawn": ("🚫", "withdrawn"),
    "": ("❔", "no receipt"),
}
RECEIPT_ITEM_ICONS = {
    "created": "✅", "updated": "✅", "exists": "✅", "needs_match": "⚠️", "invalid": "⚠️", "stale": "⚠️", "duplicate": "⚠️",
    "skipped": "⏭️", "error": "❌",
}
DONE_ITEM_STATUSES = ("created", "updated", "exists")  # receipt item statuses meaning it went through


def receipt_beside(package_file: Path) -> Path:
    return package_file.with_name(f"{package_file.stem}.receipt.json")


def load_outbox_entries(outbox: Path) -> dict[str, list[dict]]:
    """Every package by state (see OUTBOX_STATES), newest first, with its receipt if it has one."""
    entries: dict[str, list[dict]] = {state: [] for state in OUTBOX_STATES}
    files = [("waiting", path) for path in pending_outbox_files(outbox)]
    for folder in (*HANDOFF_RESULT_DIRS, WITHDRAWN_DIR):
        files += [
            (folder, path) for path in (outbox / folder).glob("*.json")
            if not path.name.startswith(".") and not path.name.endswith(".receipt.json")
        ]
    for state, path in files:
        entry = {"path": path, "package": None, "error": "", "receipt": None}
        try:
            entry["package"] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            entry["error"] = f"This file can't be read as a package: {exc}"
        if state != "waiting" and receipt_beside(path).is_file():
            try:
                entry["receipt"] = json.loads(receipt_beside(path).read_text())
            except (OSError, ValueError):
                pass
        stamp = (entry["receipt"] or {}).get("processed_at") or (entry["package"] or {}).get("created")
        try:
            when = datetime.datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            when = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        entry["when"] = when if when.tzinfo else when.astimezone()
        entries[state].append(entry)
    for group in entries.values():
        group.sort(key=lambda entry: entry["when"], reverse=True)
    return entries


def move_outbox_package(path_text: str, destination: str, action: str) -> None:
    """Move a package (and its receipt) for the Actions page buttons; `destination` is '' for the outbox itself.

    Going back to the outbox drops the old receipt first, so a connector that picks the package up at once
    can't have its fresh receipt removed. Withdrawing keeps the receipt beside the package for the record.
    """
    path = Path(path_text)
    if not path.is_file():
        st.session_state["actions_notice"] = f"`{path.name}` has already moved; the list is refreshed."
        return
    target_dir = get_outbox_dir() / destination if destination else get_outbox_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    receipt = receipt_beside(path)
    if receipt.is_file():
        if destination:
            receipt.replace(target_dir / receipt.name)
        else:
            receipt.unlink()
    path.replace(target_dir / path.name)
    st.session_state["actions_notice"] = f"{action} `{path.name}`."


def describe_candidate(candidate: dict) -> str:
    """A receipt's candidate record: its name linked to it when the connector gave a link, plus its id and details."""
    name, url = candidate.get("name", "?"), str(candidate.get("url") or "")
    text = f"[{name} ↗]({url})" if url.startswith("https://") else f"{name} (`{candidate.get('id', '?')}`)"
    return text + (f" ({candidate['detail']})" if candidate.get("detail") else "")


def create_anyway_and_retry(path_text: str, index: int) -> None:
    """Mark a held duplicate task Create anyway (a v2 field) and send its package back to the outbox."""
    change_item_and_retry(path_text, index, lambda item: item.update(allow_duplicate=True), "Create anyway")


def overwrite_anyway_and_retry(path_text: str, index: int) -> None:
    """Drop a held Next Step's `previous`, so the connector overwrites whatever is there now, and retry."""
    change_item_and_retry(path_text, index, lambda item: item.pop("previous", None), "Overwrite anyway")


def change_item_and_retry(path_text: str, index: int, change, action: str) -> None:
    """Change one item of a failed package (validated, atomic) and send the package back to the outbox."""
    path = Path(path_text)
    if not path.is_file():
        st.session_state["actions_notice"] = f"`{path.name}` has already moved; the list is refreshed."
        return
    package = json.loads(path.read_text())
    change(package["items"][index])
    package["schema"] = HANDOFF_SCHEMA_V2_ID  # v2 is v1 plus update items, mixed packages and Create anyway
    try:
        validate_handoff_package(package)
    except Exception as exc:
        st.session_state["actions_notice"] = f"Not changed: the package wouldn't be valid ({exc})."
        return
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(package, indent=2, ensure_ascii=False))
    temp.replace(path)
    move_outbox_package(path_text, "", f"{action}: sent back to the outbox:")


def delete_withdrawn_packages(path_texts: list[str]) -> None:
    """Delete withdrawn packages (and their receipts) for good. Only files in the withdrawn folder are touched."""
    withdrawn = (get_outbox_dir() / WITHDRAWN_DIR).resolve()
    deleted = 0
    for path_text in path_texts:
        path = Path(path_text)
        if path.resolve().parent != withdrawn or not path.is_file():
            continue
        receipt_beside(path).unlink(missing_ok=True)
        path.unlink()
        deleted += 1
    st.session_state["actions_notice"] = (
        f"Deleted `{Path(path_texts[0]).name}`." if deleted == 1 and len(path_texts) == 1
        else f"Deleted {deleted} withdrawn package{'s' if deleted != 1 else ''}."
    )


def outbox_entry_matches(entry: dict, query: str) -> bool:
    words = query.lower().split()
    text = json.dumps([entry["package"], entry["receipt"], entry["path"].name, entry["error"]]).lower()
    return all(word in text for word in words)


def render_outbox_entry(entry: dict, state: str) -> None:
    """One package on the Actions page: what it holds, where it came from, and the connector's result per item."""
    package = entry["package"] if isinstance(entry["package"], dict) else {}
    receipt = entry["receipt"] or {}
    items = [item for item in package.get("items", []) if isinstance(item, dict)]
    source = package.get("source") or {}
    prompt_name = (source.get("prompt") or {}).get("name", "")
    status = state if state in ("waiting", "withdrawn") else receipt.get("status", "")
    icon, status_text = PACKAGE_STATUS.get(status, PACKAGE_STATUS[""])
    kind = {"note": "note", "task": "task", "email": "email", "update": "update"}.get(package.get("item_type"), "item")
    flagged = sum(1 for index, item in enumerate(items) if item_duplicate_matches(entry, index, item)) \
        if state in DUPLICATE_CHECK_STATES else 0
    label = (
        f"{icon} {format_when(entry['when'])} · {len(items)} {kind}{'' if len(items) == 1 else 's'}"
        + (f" · Agent:{prompt_name}" if prompt_name else "")
        + f" · {status_text}"
        + (f" · ⚠ {flagged} possible duplicate{'s' if flagged != 1 else ''}" if flagged else "")
    )
    with st.expander(label, expanded=state == "failed"):
        details = [f"File `{entry['path'].name}`"]
        if package.get("id"):
            details.append(f"package id `{package['id']}`")
        if source.get("library"):
            details.append(f"from **{source['library']}**")
        if source.get("model"):
            details.append(f"model `{source['model']}`")
        st.caption(" · ".join(details))
        if source.get("question"):
            st.caption(f"Question: {source['question']}")
        notes = [note.get("file", "") for note in source.get("notes", []) if isinstance(note, dict)]
        if notes:
            st.caption("Source notes: " + ", ".join(f"`{Path(note).name}`" for note in notes))

        if entry["error"] and not receipt.get("error"):  # the connector's receipt usually says the same thing
            st.error(entry["error"])
        if state == "withdrawn":
            st.info("Withdrawn: connectors won't see it. **Restore** sends it back to the outbox.")
        if receipt:
            receipt_status = receipt.get("status", "")
            message = (
                f"**{str(receipt.get('connector', 'Connector')).capitalize()}** ({receipt.get('target', '?')}): "
                f"{RECEIPT_STATUS_TEXT.get(receipt_status, receipt_status)}."
            )
            if receipt.get("error"):
                message += f" {receipt['error']}"
            (st.success if receipt_status == "done" else st.warning)(message)
        elif state == "withdrawn":
            pass
        elif state == "waiting":
            st.info("Waiting for a connector to pick this up.")
        else:
            st.info("No receipt yet.")

        results = {r.get("index"): r for r in receipt.get("items", []) if isinstance(r, dict)}
        editable = editable_item_indexes(entry, state)
        for index, item in enumerate(items):
            result = results.get(index) or {}
            try:
                line = describe_staged_item(item)
            except (KeyError, TypeError):
                line = f"**{staged_item_title(item)}**"
            icon = RECEIPT_ITEM_ICONS.get(result.get("status"), "⏳" if state == "waiting" else "")
            st.markdown(f"{icon} {link_opportunity_names(line)}")
            if state in DUPLICATE_CHECK_STATES:
                if links := item_salesforce_links(item):
                    st.caption("Check in Salesforce: " + links)
                for match in item_duplicate_matches(entry, index, item):
                    st.markdown(describe_duplicate_match(match))
            notes_line = []
            if result.get("record_id"):
                notes_line.append(f"{result.get('object', 'Record')} `{result['record_id']}`")
            notes_line += receipt_links(result)
            if result.get("message"):
                notes_line.append(result["message"])
            candidates = ", ".join(describe_candidate(c) for c in result.get("candidates", []) if isinstance(c, dict))
            if candidates:
                notes_line.append(
                    f"{'Looks like' if result.get('status') == 'duplicate' else 'Possible matches'}: {candidates}"
                )
            if notes_line:
                st.caption(" · ".join(notes_line))
            if index in editable:
                render_reassess_panel(entry, index, f"{state}_{entry['path'].name}")
            if state == "failed" and result.get("status") == "duplicate":
                st.button(
                    "Create anyway and retry",
                    key=f"actions_create_anyway_{state}_{entry['path'].name}_{index}",
                    icon=":material/content_copy:",
                    on_click=create_anyway_and_retry,
                    args=(str(entry["path"]), index),
                    help="Create this task even though it looks like an open one, and send the package back to the "
                    "outbox. Anything already created isn't created twice.",
                )
            if state == "failed" and result.get("status") == "stale":
                st.button(
                    "Overwrite anyway and retry",
                    key=f"actions_overwrite_anyway_{state}_{entry['path'].name}_{index}",
                    icon=":material/swap_horiz:",
                    on_click=overwrite_anyway_and_retry,
                    args=(str(entry["path"]), index),
                    help="Replace whatever the Next Step says now with this one, and send the package back to the "
                    "outbox. Anything already done isn't done twice.",
                )
        if state == "failed":
            st.caption(
                "If a record needs choosing, choose it in the connector (it then retries by itself). After a "
                "temporary error, **Retry** sends the package back for the connector's next run; anything already "
                "created isn't created twice."
            )
        render_outbox_entry_actions(entry, state)


def render_outbox_entry_actions(entry: dict, state: str) -> None:
    """Withdraw (waiting, failed), Retry (failed), and Restore and Delete (withdrawn) buttons for one package."""
    path, key = str(entry["path"]), f"{state}_{entry['path'].name}"
    unreadable = bool(entry["error"] or (entry["receipt"] or {}).get("error"))
    with st.container(horizontal=True):
        if state == "failed":
            st.button(
                "Retry",
                icon=":material/replay:",
                key=f"actions_retry_{key}",
                on_click=move_outbox_package,
                args=(path, "", "Sent back to the outbox:"),
                disabled=unreadable,
                help=(
                    "This file itself is invalid, so retrying won't help. Withdraw it and send the drafts again."
                    if unreadable else "Move it back to the outbox for the connector's next run."
                ),
            )
        if state in ("waiting", "failed"):
            with st.popover("Withdraw", icon=":material/block:", help="Take it out of the outbox so no connector acts on it"):
                st.markdown(
                    "Withdraw this package? Connectors won't see it. Nothing already created in the other system "
                    "is undone. You can restore it from the **Withdrawn** tab."
                )
                st.button(
                    "Withdraw",
                    key=f"actions_withdraw_{key}",
                    type="primary",
                    on_click=move_outbox_package,
                    args=(path, WITHDRAWN_DIR, "Withdrew"),
                )
        if state == "withdrawn":
            st.button(
                "Restore",
                icon=":material/undo:",
                key=f"actions_restore_{key}",
                on_click=move_outbox_package,
                args=(path, "", "Restored to the outbox:"),
                help="Send it back to the outbox for a connector to process.",
            )
            with st.popover("Delete", icon=":material/delete:", help="Delete this package for good"):
                st.markdown("Delete this package and its receipt for good? This can't be undone.")
                st.button(
                    "Delete",
                    key=f"actions_delete_{key}",
                    type="primary",
                    on_click=delete_withdrawn_packages,
                    args=([path],),
                )
        editable = editable_item_indexes(entry, state)
        if editable:
            editing = st.toggle(
                "Edit",
                key=f"actions_editing_{key}",
                help="Change the text before a connector acts on it, e.g. add a marker like [TEST] to find it easily.",
            )
    if editable and editing:
        render_package_editor(entry, key, editable, state)


EDITABLE_OUTBOX_STATES = ("waiting", "failed", "withdrawn")


def editable_item_indexes(entry: dict, state: str) -> list[int]:
    """Items that can still be edited: not processed yet, or (for failed packages) not created yet."""
    package = entry["package"]
    if state not in EDITABLE_OUTBOX_STATES or entry["error"] or not isinstance(package, dict):
        return []
    created = {
        r.get("index") for r in (entry["receipt"] or {}).get("items", [])
        if isinstance(r, dict) and r.get("status") in DONE_ITEM_STATUSES
    }
    return [i for i, item in enumerate(package.get("items", [])) if isinstance(item, dict) and i not in created]


def package_edit_key(entry_key: str, index: int, field: str) -> str:
    return f"actions_edit_{entry_key}_{index}_{field}"


def render_package_editor(entry: dict, entry_key: str, indexes: list[int], state: str) -> None:
    """Form to edit a package's items in place; saving rewrites the file atomically with the same package id."""
    items = entry["package"]["items"]
    if state == "failed" and len(indexes) < len(items):
        st.caption("Records already created aren't shown: editing them wouldn't change what's in the other system.")
    with st.form(f"actions_edit_form_{entry_key}", border=True):
        for index in indexes:
            item = items[index]
            key = lambda field: package_edit_key(entry_key, index, field)  # noqa: E731
            item_type = item.get("type", "")
            st.markdown(f"**{ {'note': 'Note', 'task': 'Task', 'email': 'Email', 'update': 'Next Step'}.get(item_type, 'Item') } {index + 1}**")
            if item_type == "update":
                st.text_area("New Next Step", value=item.get("value", ""), key=key("value"), height=80,
                             max_chars=NEXT_STEP_LIMIT, help=f"At most {NEXT_STEP_LIMIT} characters.")
            elif item_type == "note":
                st.text_input("Title", value=item.get("title", ""), key=key("title"))
                st.text_area("Body", value=item.get("body", ""), key=key("body"), height=140)
            elif item_type == "task":
                st.text_input("Subject", value=item.get("subject", ""), key=key("subject"))
                st.text_area("Description", value=item.get("description", ""), key=key("description"), height=80)
                due_col, priority_col, owner_col = st.columns(3)
                due = item.get("due_date", "")
                due_col.date_input(
                    "Due date",
                    value=datetime.date.fromisoformat(due) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", due) else None,
                    key=key("due_date"),
                    format="YYYY-MM-DD",
                )
                priority = item.get("priority", "")
                priority_col.selectbox(
                    "Priority",
                    TASK_PRIORITIES,
                    index=TASK_PRIORITIES.index(priority) if priority in TASK_PRIORITIES else 0,
                    format_func=lambda p: p.capitalize() or "Not set",
                    key=key("priority"),
                )
                owner_col.text_input("Owner", value=item.get("owner", ""), key=key("owner"), placeholder="Not given")
            else:
                st.text_input("To", value=", ".join(item.get("to", [])), key=key("to"), help="Comma-separated.")
                st.text_input("Subject", value=item.get("subject", ""), key=key("subject"))
                st.text_area("Body", value=item.get("body", ""), key=key("body"), height=140)
            related = item.get("related_to") or {}
            name_col, kind_col = st.columns([2, 1])
            name_col.text_input("Related to", value=related.get("name", ""), key=key("related_name"))
            kind = related.get("kind", "")
            kind_col.selectbox(
                "Kind",
                RELATED_KINDS,
                index=RELATED_KINDS.index(kind) if kind in RELATED_KINDS else 0,
                format_func=lambda k: k.capitalize() or "Not sure",
                key=key("related_kind"),
            )
        st.form_submit_button(
            "Save changes",
            type="primary",
            icon=":material/save:",
            on_click=save_package_edits,
            args=(str(entry["path"]), entry_key, indexes),
        )


def save_package_edits(path_text: str, entry_key: str, indexes: list[int]) -> None:
    """Write the edited items back to the package file (validated, atomic, same id), then close the editor."""
    path = Path(path_text)
    if not path.is_file():
        st.session_state["actions_notice"] = f"`{path.name}` has moved (a connector may have picked it up); nothing saved."
        return
    package = json.loads(path.read_text())
    problems = []
    for index in indexes:
        item = package["items"][index]
        value = lambda field, default: st.session_state.get(package_edit_key(entry_key, index, field), default)  # noqa: E731
        related = item.get("related_to") or {}
        edited = {
            **item,
            "related_to": {
                "name": value("related_name", related.get("name", "")),
                "kind": value("related_kind", related.get("kind", "")),
            },
        }
        for field in ("title", "body", "subject", "description", "priority", "owner", "value"):
            if field in item:
                edited[field] = value(field, item[field])
        if "due_date" in item:
            due = value("due_date", None)
            edited["due_date"] = due.isoformat() if isinstance(due, datetime.date) else ""
        if "to" in item:
            to = value("to", ", ".join(item["to"]))
            edited["to"] = to.split(",") if isinstance(to, str) else to
        cleaned = normalise_draft_item(item["type"], edited)
        if item["type"] == "update":
            if not cleaned["value"]:
                problems.append(f"Item {index + 1} needs a Next Step.")
        elif not (cleaned.get("subject") or cleaned.get("title")):
            problems.append(f"Item {index + 1} needs a {'title' if item['type'] == 'note' else 'subject'}.")
        package["items"][index] = cleaned
    if problems:
        st.session_state["actions_notice"] = "Not saved: " + " ".join(problems)
        return
    try:
        validate_handoff_package(package)
    except Exception as exc:
        st.session_state["actions_notice"] = f"Not saved: the package wouldn't be valid ({exc})."
        return
    temp = path.with_name(f".{path.name}.tmp")  # connectors skip dotfiles, so they never see a half-written file
    temp.write_text(json.dumps(package, indent=2, ensure_ascii=False))
    if not path.is_file():  # picked up while we were writing: don't recreate it
        temp.unlink()
        st.session_state["actions_notice"] = f"`{path.name}` has moved (a connector may have picked it up); nothing saved."
        return
    temp.replace(path)
    for widget_key in [k for k in st.session_state if str(k).startswith(f"actions_edit_{entry_key}_")]:
        del st.session_state[widget_key]
    st.session_state[f"actions_editing_{entry_key}"] = False
    st.session_state["actions_notice"] = f"Saved changes to `{path.name}`."


# --- Agent Semantic Fixes and Re-assess on the Actions page ---------------------
# Facts that repair what Agents send, e.g. "Joseph Oliver is known as J-O", kept in
# the prompt library. Re-assess asks the model to correct one outbox item with them, any extra context typed
# for it and the package's source notes; nothing changes until Apply.
REASSESS_FIELDS = {
    "task": ("subject", "description", "due_date", "priority", "owner", "related_to"),
    "note": ("title", "body", "related_to"),
    "email": ("to", "subject", "body", "related_to"),
    "update": ("value", "related_to"),
}
REASSESS_FIELD_NAMES = {"value": "Next Step", "to": "To", "due_date": "Due date", "related_to": "Related to"}
REASSESS_NOTES_CHARS = 8000  # source note text sent with a Re-assess, shared between the notes
REASSESS_KEY = "reassess"  # session state: {entry key and item index: the last Re-assess's proposal}
OFFER_FIX_KEY = "offer_semantic_fix"  # context just applied, to offer as a new Agent Semantic Fix


def semantic_fixes(library: dict | None = None) -> list[dict]:
    return (library or load_prompt_library()).get("semantic_fixes", [])


def add_semantic_fix(text: str) -> None:
    text = text.strip()
    if not text:
        return
    library = load_prompt_library()
    library.setdefault("semantic_fixes", []).append({"id": uuid.uuid4().hex, "text": text, "added": now_iso()})
    save_prompt_library(library)


def add_semantic_fix_from_form() -> None:
    add_semantic_fix(str(st.session_state.get("new_semantic_fix", "")))


def save_semantic_fix(fix_id: str) -> None:
    """Save an edited fix; emptying the box keeps the old text (use ✕ to remove one)."""
    text = str(st.session_state.get(f"semantic_fix_{fix_id}", "")).strip()
    library = load_prompt_library()
    fix = next((f for f in library.get("semantic_fixes", []) if f["id"] == fix_id), None)
    if fix and text and text != fix["text"]:
        fix.update(text=text, edited=now_iso())
        save_prompt_library(library)
        mark_text_saved(f"semantic_fix_{fix_id}")


def delete_semantic_fix(fix_id: str) -> None:
    library = load_prompt_library()
    library["semantic_fixes"] = [f for f in library.get("semantic_fixes", []) if f["id"] != fix_id]
    save_prompt_library(library)


def format_fix_date(iso_time: str) -> str:
    try:
        return datetime.datetime.fromisoformat(iso_time).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return "?"


def render_semantic_fixes_section(library: dict) -> None:
    """Prompt library: the Agent Semantic Fixes, each a big editable box with the date it was added."""
    fixes = semantic_fixes(library)
    st.html(f"<style>.st-key-prompt_page_fixes_open details {{ border: 1px solid rgb({AGENT_TINT}) !important; }}</style>")
    with st.expander(
        f"Agent Semantic Fixes · {len(fixes)}",
        expanded=st.query_params.get("section") == "fixes",
        key="prompt_page_fixes_open",
        icon=":material/build:",
    ):
        st.caption(
            "Facts that repair what Agents send, e.g. *Joseph Oliver is known as J-O*. "
            "**Re-assess** on the Actions page uses them all to correct an item a connector couldn't process. "
            "Applying a fix there offers to add its context here. Saved as you type."
        )
        for fix in fixes:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(f"Added {format_fix_date(fix.get('added'))}"
                           + (f" · edited {format_fix_date(fix['edited'])}" if fix.get("edited") else ""))
                st.button(":material/close:", key=f"semantic_fix_delete_{fix['id']}", type="tertiary",
                          on_click=delete_semantic_fix, args=(fix["id"],), help="Remove this fix")
            st.text_area(
                "Fix", value=fix["text"], key=f"semantic_fix_{fix['id']}", height=120, label_visibility="collapsed",
                on_change=save_semantic_fix, args=(fix["id"],),
            )
            render_saved_note(f"semantic_fix_{fix['id']}")
        with st.form("add_semantic_fix_form", clear_on_submit=True, border=False):
            st.text_area("New fix", key="new_semantic_fix", height=120,
                         placeholder="e.g. Joseph Oliver is known as J-O")
            st.form_submit_button("Add fix", icon=":material/add:", on_click=add_semantic_fix_from_form)


def source_notes_text(package: dict, budget: int = REASSESS_NOTES_CHARS) -> str:
    """The package's source notes re-read from disk, each cut to an equal share of `budget` characters."""
    paths = [Path(n["path"]) for n in (package.get("source") or {}).get("notes", [])
             if isinstance(n, dict) and n.get("path") and Path(n["path"]).is_file()]
    share = budget // max(len(paths), 1)
    parts = []
    for path in paths:
        try:
            text = path.read_text(errors="replace").strip()
        except OSError:
            continue
        parts.append(f"--- {path.name} ---\n" + (text[:share].rstrip() + " […]" if len(text) > share else text))
    return "\n\n".join(parts)


def reassess_schema(item: dict) -> dict:
    """JSON schema for a Re-assess reply: the item's editable fields, plus `reason`."""
    types = {"to": {"type": "array", "items": {"type": "string"}},
             "related_to": {"type": "object", "properties": {"name": {"type": "string"}, "kind": {"type": "string"}},
                            "required": ["name", "kind"]}}
    fields = REASSESS_FIELDS.get(item.get("type"), ())
    properties = {field: types.get(field, {"type": "string"}) for field in fields}
    properties["reason"] = {"type": "string"}
    return {"type": "object", "properties": properties, "required": [*fields, "reason"]}


def reassess_messages(package: dict, index: int, receipt: dict | None, context: str, fixes: list[dict],
                      item: dict | None = None) -> list[dict]:
    item = item or package["items"][index]
    fields = REASSESS_FIELDS.get(item.get("type"), ())
    result = next((r for r in (receipt or {}).get("items", []) if isinstance(r, dict) and r.get("index") == index), {})
    problem = " ".join(p for p in (str(result.get("message") or ""), ", ".join(
        str(c.get("name", "")) for c in result.get("candidates", []) if isinstance(c, dict))) if p)
    kind = ITEM_KIND_NAMES.get(item.get("type"), "Item").lower()
    notes = source_notes_text(package)
    parts = [f"Today is {datetime.date.today():%A %Y-%m-%d}. An Agent drafted this {kind} for Salesforce from my notes."]
    if problem:
        parts.append(f"The connector couldn't process it: {problem}")
    if fixes:
        parts.append("Facts to apply:\n" + "\n".join(f"- {f['text']}" for f in fixes))
    if context.strip():
        parts.append(f"More context for this {kind}:\n{context.strip()}")
    if notes:
        parts.append(f"The notes it came from:\n{notes}")
    parts.append(f"The {kind} as JSON:\n" + json.dumps({f: item.get(f) for f in fields}, ensure_ascii=False))
    parts.append(
        "Return it with the same fields. Salesforce only recognises people by their full name, so wherever a "
        "person's field (owner, recipients, related to) holds a nickname, short name or alias that the facts or "
        "the context give a full name for, replace it with that full name, even if the notes use the nickname. "
        "Otherwise change only what the facts, the context or the notes show is wrong or needed to fix the "
        "problem, and keep every other field exactly as it is. Set `reason` to one short sentence on what you "
        "changed and why, or why nothing needed changing."
    )
    return [{"role": "user", "content": "\n\n".join(parts)}]


ALIAS_SCHEMA = {"type": "object", "required": ["aliases"], "properties": {"aliases": {"type": "array", "items": {
    "type": "object", "required": ["alias", "full_name"],
    "properties": {"alias": {"type": "string"}, "full_name": {"type": "string"}}}}}}
PERSON_FIELDS = ("owner", "to")  # fields holding people, where a known nickname is replaced by the full name


def extract_aliases(texts: list[str], model_name: str, chat) -> dict[str, str]:
    """{nickname, lower case: full name} stated in the fixes and context. The model only reads them out; the
    app makes the swap, so it doesn't depend on the model judging the item (cached for the session)."""
    facts = "\n".join(f"- {t.strip()}" for t in texts if t.strip())
    if not facts:
        return {}
    cache = st.session_state.setdefault("reassess_alias_cache", {})
    key = json.dumps([facts, model_name])
    if key not in cache:
        reply, _usage = chat(
            [{"role": "user", "content": (
                "List every person these facts give a nickname, short name or alias for, with their full name. "
                "Only pairs the facts state; an empty list if there are none.\n\n" + facts)}],
            model_name, max_tokens=512, temperature=0.0, response_format=ALIAS_SCHEMA, think=False,
        )
        try:
            pairs = json.loads(reply).get("aliases", [])
        except (ValueError, AttributeError):
            pairs = []
        cache[key] = {
            str(p.get("alias", "")).strip().lower(): str(p.get("full_name", "")).strip()
            for p in pairs if isinstance(p, dict) and str(p.get("alias", "")).strip() and str(p.get("full_name", "")).strip()
            and str(p.get("alias", "")).strip().lower() != str(p.get("full_name", "")).strip().lower()
        }
    return cache[key]


def apply_aliases(item: dict, aliases: dict[str, str]) -> dict:
    """The item with each person field that is exactly a known nickname replaced by the full name."""
    resolved = dict(item)
    if isinstance(resolved.get("owner"), str):
        resolved["owner"] = aliases.get(resolved["owner"].strip().lower(), resolved["owner"])
    if isinstance(resolved.get("to"), list):
        resolved["to"] = [aliases.get(str(r).strip().lower(), r) for r in resolved["to"]]
    return resolved


def reassess_item(entry: dict, index: int, context: str, model_name: str, chat=None) -> dict:
    """Dry run: the corrected item and what changed (each marked as a name fix or the model's). Nothing is written.

    Known nicknames in person fields are swapped for full names first (see extract_aliases); the model then
    re-assesses the item with those swaps in place, and can't undo them."""
    chat = chat or chat_with_ollama
    package = entry["package"]
    item = package["items"][index]
    fixes = semantic_fixes()
    aliases = extract_aliases([f["text"] for f in fixes] + [context], model_name, chat)
    resolved = apply_aliases(item, aliases)
    reply, _usage = chat(reassess_messages(package, index, entry["receipt"], context, fixes, resolved), model_name,
                         max_tokens=1024, temperature=0.1, num_ctx=FOLLOW_UP_NUM_CTX,
                         response_format=reassess_schema(item), think=False)
    try:
        data = json.loads(reply)
    except ValueError as exc:
        raise RuntimeError(f"`{model_name}` didn't return complete JSON; try again or pick a larger model.") from exc
    fields = REASSESS_FIELDS.get(item.get("type"), ())
    cleaned = normalise_draft_item(item["type"], {**resolved, **{f: data[f] for f in fields if f in data}})
    proposal = {**item, **{f: cleaned[f] for f in fields if f in cleaned}}  # keep any other keys as they were
    named = [f for f in PERSON_FIELDS if f in item and resolved.get(f) != item.get(f)]
    for field in named:
        proposal[field] = resolved[field]
    changes = [(f, item.get(f), proposal.get(f)) for f in fields if proposal.get(f) != item.get(f)]
    return {"original": item, "proposal": proposal, "changes": changes, "named": named,
            "reason": str(data.get("reason", "")).strip(), "context": context, "model": model_name}


def describe_reassess_value(field: str, value) -> str:
    if field == "related_to":
        value = (value or {}).get("name", "") + (f" ({value['kind']})" if (value or {}).get("kind") else "")
    elif field == "to":
        value = ", ".join(value or [])
    return f"`{value}`" if value else "*empty*"


def run_reassess(entry: dict, index: int, entry_key: str) -> None:
    context = str(st.session_state.get(f"reassess_context_{entry_key}_{index}", ""))
    model = st.session_state.get("duplicate_check_model") or ""
    try:
        with st.spinner(f"Re-assessing with `{model}`…"):
            outcome = reassess_item(entry, index, context, model)
    except Exception as exc:
        outcome = {"error": format_exception_for_user(exc)}
    st.session_state.setdefault(REASSESS_KEY, {})[f"{entry_key}_{index}"] = outcome


def apply_reassessment(path_text: str, index: int, entry_key: str) -> None:
    """Write the proposal into the package, send it back to the outbox, then offer the context as a fix."""
    outcome = st.session_state.get(REASSESS_KEY, {}).pop(f"{entry_key}_{index}", None)
    if not outcome or "proposal" not in outcome:
        return
    picked = [f for f, _old, _new in outcome["changes"] if st.session_state.get(f"reassess_pick_{entry_key}_{index}_{f}", True)]
    if not picked:
        st.session_state["actions_notice"] = "Nothing applied: no change was ticked."
        return
    proposal = {**outcome["original"], **{f: outcome["proposal"][f] for f in picked}}

    def change(item: dict) -> None:
        item.clear()
        item.update(proposal)

    change_item_and_retry(path_text, index, change, "Re-assessed")
    st.session_state[f"reassess_open_{entry_key}_{index}"] = False
    applied = str(st.session_state.get("actions_notice", "")).startswith("Re-assessed")
    if applied and outcome["context"].strip():
        st.session_state[OFFER_FIX_KEY] = outcome["context"].strip()


def discard_reassessment(entry_key: str, index: int) -> None:
    st.session_state.get(REASSESS_KEY, {}).pop(f"{entry_key}_{index}", None)


def render_reassess_panel(entry: dict, index: int, entry_key: str) -> None:
    """Inline under an editable item: extra context, Re-assess (dry run), then the proposed changes to apply."""
    slot = f"{entry_key}_{index}"
    if not st.toggle("Fix with context", key=f"reassess_open_{slot}",
                     help="Add context, e.g. who a nickname is, and have the model re-assess this item."):
        return
    with st.container(border=True):
        st.text_area("Extra context", key=f"reassess_context_{slot}", height=80,
                     placeholder="e.g. Joseph Oliver is known as J-O")
        fixes = len(semantic_fixes())
        model = st.session_state.get("duplicate_check_model") or ""
        st.caption(
            f"Re-assess sends the model this item, the connector's message, your context, the "
            f"[{plural(fixes, 'Agent Semantic Fix')}]({PROMPT_LIBRARY_PAGE_URL}&section=fixes) and the package's "
            f"source notes. It uses `{model}`, the **Model** picked above. Nothing changes until you apply it."
        )
        if st.button("Re-assess", icon=":material/psychology:", key=f"reassess_run_{slot}", disabled=not model):
            run_reassess(entry, index, entry_key)
        outcome = st.session_state.get(REASSESS_KEY, {}).get(slot)
        if not outcome:
            return
        if outcome.get("error"):
            st.error(f"Re-assess failed: {outcome['error']}")
            return
        if outcome["reason"]:
            st.caption(f"**{outcome['model']}:** {outcome['reason']}")
        if not outcome["changes"]:
            st.info("No change proposed. Add more context and re-assess, or edit the item yourself.")
            return
        st.caption("Untick any change you don't want; only ticked ones are applied.")
        for field, old, new in outcome["changes"]:
            name = REASSESS_FIELD_NAMES.get(field, field.replace("_", " ").capitalize())
            source = " · full name from the fixes" if field in outcome.get("named", []) else ""
            st.checkbox(f"**{name}:** ~~{describe_reassess_value(field, old)}~~ → "
                        f"{describe_reassess_value(field, new)}{source}", value=True, key=f"reassess_pick_{slot}_{field}")
        with st.container(horizontal=True):
            st.button("Apply and retry", type="primary", icon=":material/check:", key=f"reassess_apply_{slot}",
                      on_click=apply_reassessment, args=(str(entry["path"]), index, entry_key),
                      help="Save the ticked changes in the package and send it back to the outbox for the connector's "
                      "next run (a withdrawn package goes back too). Anything already created isn't created twice.")
            st.button("Discard", key=f"reassess_discard_{slot}", on_click=discard_reassessment, args=(entry_key, index))


def add_offered_fix() -> None:
    add_semantic_fix(str(st.session_state.get("offer_fix_text", "")))
    st.session_state["actions_notice"] = "Added to Agent Semantic Fixes."


@st.dialog("Add to Agent Semantic Fixes?")
def offer_semantic_fix_dialog(text: str) -> None:
    st.markdown("The fix is applied. Keep its context as an **Agent Semantic Fix**, so Re-assess uses it next time?")
    st.text_area("Fix", value=text, key="offer_fix_text", height=120)
    with st.container(horizontal=True):
        # Saved in a callback, so it's kept even if the page reruns in full; then a full rerun closes the dialog.
        add = st.button("Add to fixes", type="primary", icon=":material/add:", key="offer_fix_add",
                        on_click=add_offered_fix)
        skip = st.button("Not now", key="offer_fix_skip")
    if add or skip:
        st.rerun()


# --- Duplicate check on the Actions page ----------------------------------------
DUPLICATE_CHECK_KEY = "outbox_duplicate_check"  # the last check's results, for this session
DUPLICATE_CHECK_STATES = ("waiting", "withdrawn")  # the items checked; any state can be what they repeat
DUPLICATE_CHECK_MAX_TOKENS = 2048  # room for a reasoning model (e.g. Qwen3) to think before it answers
DUPLICATE_CANDIDATE_LIMIT = 12  # existing items shown to the model per checked item, same record first
# A verdict on every candidate, not just a list of matches: small models then flag far fewer false duplicates.
DUPLICATE_CHECK_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"number": {"type": "integer"}, "same": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["number", "same", "reason"],
    }}},
    "required": ["verdicts"],
}
ITEM_KIND_NAMES = {"task": "Task", "note": "Note", "email": "Email", "update": "Next Step"}


def outbox_item_key(entry: dict, index: int) -> str:
    return f"{entry['path'].name}#{index}"


def outbox_item_text(item: dict) -> str:
    """What an item says, for the model: its kind, title and the start of its body."""
    body = item.get("description") or item.get("body") or ""
    text = f"{ITEM_KIND_NAMES.get(item.get('type'), 'Item')}: {staged_item_title(item)}"
    return text + (f" — {truncate_text(body, 300)}" if body else "")


def item_title_words(item: dict) -> set[str]:
    """Key words of an item's title (a Next Step's value), as task subjects are compared."""
    return task_subject_words(item.get("value", "") if item.get("type") == "update" else staged_item_title(item))


def item_record_name(item: dict) -> str:
    return str((item.get("related_to") or {}).get("name") or "").strip().lower()


def outbox_items(entries: dict[str, list[dict]]) -> list[dict]:
    """Every readable item in the outbox, in every state, with where it is."""
    found = []
    for state, group in entries.items():
        for entry in group:
            package = entry["package"] if isinstance(entry["package"], dict) else {}
            for index, item in enumerate(package.get("items", [])):
                if isinstance(item, dict) and item.get("type"):
                    found.append({"key": outbox_item_key(entry, index), "state": state, "item": item,
                                  "when": entry["when"], "text": outbox_item_text(item)})
    return found


def duplicate_candidates(checked: dict, others: list[dict], open_tasks: list[dict]) -> list[dict]:
    """What a checked item might repeat: items of the same kind on the same record, or, when either has no
    record, sharing a key word; tasks also against my open tasks in Salesforce. Same record first, newest first."""
    item, record = checked["item"], item_record_name(checked["item"])
    words = item_title_words(item)

    def related(other_record: str, other_words: set[str]) -> int:
        if record and other_record:
            return 2 if record == other_record else 0
        return 1 if words & other_words else 0

    found = []
    for other in others:
        if other["key"] == checked["key"] or other["item"].get("type") != item.get("type"):
            continue
        if score := related(item_record_name(other["item"]), item_title_words(other["item"])):
            found.append({"kind": "outbox", "score": score, "when": other["when"], "key": other["key"],
                          "state": other["state"], "text": other["text"]})
    if item.get("type") == "task":
        for task in open_tasks:
            if score := related(str(task.get("related") or "").strip().lower(), task_subject_words(task["subject"])):
                found.append({"kind": "salesforce", "score": score, "when": None, "task": task,
                              "text": f"Task: {task['subject']}"})
    found.sort(key=lambda c: (-c["score"], -(c["when"].timestamp() if c["when"] else 0)))
    return found[:DUPLICATE_CANDIDATE_LIMIT]


def duplicate_check_messages(checked: dict, candidates: list[dict]) -> list[dict]:
    record = (checked["item"].get("related_to") or {}).get("name") or "no record"
    lines = "\n".join(f"{n}. {c['text']}" + (" (already in Salesforce)" if c["kind"] == "salesforce" else "")
                      for n, c in enumerate(candidates, 1))
    return [{"role": "user", "content": (
        "You check a CRM outbox for duplicates. Below is a NEW item and a numbered list of EXISTING items.\n"
        "An existing item is a duplicate only when it is the same action or says the same thing as the new item, "
        "even in different words (e.g. 'Send revised pricing' and 'Email Jo the updated price list'). Different "
        "actions on the same deal are NOT duplicates (e.g. 'Book kickoff workshop' and 'Send revised pricing'), nor "
        "is a follow-up to an earlier action unless it repeats it.\n"
        "For every existing item, say whether it is the same action (same: true or false) with a short reason.\n\n"
        f"NEW item (on {record}):\n{checked['text']}\n\nEXISTING items:\n{lines}"
    )}]


def run_duplicate_check(entries: dict[str, list[dict]], model_name: str, chat=None, on_step=None) -> dict:
    """Ask the model, for each Waiting and Withdrawn item, which of its shortlisted items it repeats.

    A pair is asked about once. Results map each item key to its matches, both ways round, so the other
    item shows the match too when it's on screen.
    """
    chat = chat or chat_with_ollama
    items = outbox_items(entries)
    open_tasks = reference_open_tasks()
    checked_items = [i for i in items if i["state"] in DUPLICATE_CHECK_STATES]
    by_key = {i["key"]: i for i in items}
    matches: dict[str, list[dict]] = {}
    asked: set[frozenset] = set()
    errors, comparisons = [], 0
    for number, checked in enumerate(checked_items, 1):
        if on_step:
            on_step(number, len(checked_items))
        candidates = [c for c in duplicate_candidates(checked, items, open_tasks)
                      if c["kind"] == "salesforce" or frozenset((checked["key"], c["key"])) not in asked]
        if not candidates:
            continue
        asked.update(frozenset((checked["key"], c["key"])) for c in candidates if c["kind"] == "outbox")
        comparisons += len(candidates)
        try:
            # Thinking off: a reasoning model judges these as well without it, and many times faster.
            reply, _usage = chat(duplicate_check_messages(checked, candidates), model_name,
                                 max_tokens=DUPLICATE_CHECK_MAX_TOKENS, temperature=0.0,
                                 response_format=DUPLICATE_CHECK_SCHEMA, think=False)
            found = [v for v in json.loads(reply).get("verdicts", []) if isinstance(v, dict) and v.get("same") is True]
        except Exception as exc:  # one bad reply shouldn't stop the rest
            errors.append(f"{checked['text'][:60]}: {exc}")
            continue
        for match in found:
            try:
                candidate = candidates[int(match["number"]) - 1]
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            reason = str(match.get("reason") or "").strip()
            if candidate["kind"] == "salesforce":
                matches.setdefault(checked["key"], []).append(
                    {"kind": "salesforce", "task": candidate["task"], "reason": reason, "text": checked["text"]})
                continue
            other = by_key[candidate["key"]]
            matches.setdefault(checked["key"], []).append(
                {"kind": "outbox", "key": other["key"], "state": other["state"], "when": other["when"],
                 "text": checked["text"], "other_text": other["text"], "reason": reason})
            matches.setdefault(other["key"], []).append(
                {"kind": "outbox", "key": checked["key"], "state": checked["state"], "when": checked["when"],
                 "text": other["text"], "other_text": checked["text"], "reason": reason})
    return {"when": datetime.datetime.now(), "model": model_name, "checked": len(checked_items),
            "comparisons": comparisons, "open_tasks": len(open_tasks), "matches": matches, "errors": errors}


def item_duplicate_matches(entry: dict, index: int, item: dict) -> list[dict]:
    """The last check's matches for this item, unless it has been edited since."""
    check = st.session_state.get(DUPLICATE_CHECK_KEY) or {}
    text = outbox_item_text(item)
    return [m for m in check.get("matches", {}).get(outbox_item_key(entry, index), []) if m["text"] == text]


def describe_duplicate_match(match: dict) -> str:
    if match["kind"] == "salesforce":
        return f"⚠ **Looks like your open Salesforce task** {describe_open_task(match['task'])} — {match['reason']}"
    where = f"{OUTBOX_STATES.get(match['state'], ('',))[0]} · {format_when(match['when'])}"
    return f"⚠ **Looks like** “{truncate_text(match['other_text'], 120)}” ({where}) — {match['reason']}"


def item_salesforce_links(item: dict) -> str:
    """Links to check an item in Salesforce: the record it's on, and my open tasks on that record."""
    name = (item.get("related_to") or {}).get("name") or ""
    record = find_reference_record(name) if name else None
    links = []
    if record and str(record.get("url") or "").startswith("https://"):
        links.append(f"[Open {record['name']} ↗]({record['url']})")
    on = name.strip().lower()
    tasks = [t for t in reference_open_tasks() if on and str(t.get("related") or "").strip().lower() == on]
    if tasks:
        shown = [f"[{truncate_text(t['subject'], 40)} ↗]({t['url']})" if str(t.get("url") or "").startswith("https://")
                 else f"“{truncate_text(t['subject'], 40)}”" for t in tasks[:5]]
        links.append(f"your open tasks on it: {', '.join(shown)}" + (f" and {len(tasks) - 5} more" if len(tasks) > 5 else ""))
    return " · ".join(links)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def describe_check_scope(entries: dict[str, list[dict]]) -> str:
    """'7 items in 3 packages (6 tasks, 1 Next Step)': what the check covers, counted the way the tabs are."""
    items = [i for i in outbox_items(entries) if i["state"] in DUPLICATE_CHECK_STATES]
    packages = sum(len(entries[state]) for state in DUPLICATE_CHECK_STATES)
    kinds: dict[str, int] = {}
    for i in items:
        name = ITEM_KIND_NAMES.get(i["item"].get("type"), "Item")
        kinds[name] = kinds.get(name, 0) + 1
    breakdown = ", ".join(plural(n, name if name == "Next Step" else name.lower()) for name, n in kinds.items())
    return f"{plural(len(items), 'item')} in {plural(packages, 'package')}" + (f" ({breakdown})" if breakdown else "")


def duplicate_check_model_options() -> tuple[bool, list[str]]:
    ok, installed = discover_ollama_models()
    return ok, list(installed if ok and installed else SUGGESTED_OLLAMA_MODELS)


def model_size(name: str) -> float:
    """Billions of parameters from a model tag such as `qwen3:30b` or `llama3.2:3b` (0 if it doesn't say)."""
    match = re.search(r"(\d+(?:\.\d+)?)b\b", name.lower())
    return float(match.group(1)) if match else 0.0


def persist_duplicate_check_model() -> None:
    save_app_settings({"duplicate_check_model": st.session_state.get("duplicate_check_model")})


def render_duplicate_check_panel(entries: dict[str, list[dict]]) -> None:
    """Actions page: a button to check Waiting and Withdrawn items for duplicates, and the last check's summary."""
    to_check = sum(1 for i in outbox_items(entries) if i["state"] in DUPLICATE_CHECK_STATES)
    ok, models = duplicate_check_model_options()
    if st.session_state.get("duplicate_check_model") not in models:
        # Until one is chosen, the largest installed model: small ones judge duplicates poorly.
        saved = load_app_settings().get("duplicate_check_model")
        st.session_state["duplicate_check_model"] = saved if saved in models else max(models, key=model_size)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            model = st.selectbox("Model", models, key="duplicate_check_model", on_change=persist_duplicate_check_model,
                                 width=220, help="The local Ollama model that judges whether two items are the same. A larger model "
                                 "(e.g. `qwen3:30b`) judges much better; a small one like `llama3.2:3b` misses "
                                 "rewordings and flags different actions.")
            clicked = st.button(
                "🔍 Check for duplicates",
                key="duplicate_check_run",
                disabled=not ok or not to_check,
                help="Compares each Waiting and Withdrawn item with the other items in the outbox (any tab) and, for "
                "tasks, your open tasks in Salesforce. Items on the same record, or sharing a word when one has no "
                "record, are shortlisted; the model then judges which say the same thing, even in other words. "
                "Nothing is changed: likely duplicates are marked ⚠ for you to withdraw or keep.",
            )
        if not ok:
            st.caption("Ollama isn't running, so the check can't run.")
        if clicked:
            bar = st.progress(0.0, text="Checking…")
            st.session_state[DUPLICATE_CHECK_KEY] = {
                **run_duplicate_check(entries, model,
                                      on_step=lambda n, total: bar.progress(n / total, text=f"Checking item {n} of {total}…")),
                "scope": describe_check_scope(entries),
            }
            bar.empty()
        check = st.session_state.get(DUPLICATE_CHECK_KEY)
        if check:
            flagged = sum(1 for i in outbox_items(entries) if i["state"] in DUPLICATE_CHECK_STATES
                          and any(m["text"] == i["text"] for m in check["matches"].get(i["key"], [])))
            st.caption(
                f"Checked {check['scope']} on the Waiting and Withdrawn tabs at "
                f"{check['when']:%H:%M} with `{check['model']}`: {check['comparisons']} comparisons, against the "
                f"outbox and your {check['open_tasks']} open Salesforce tasks. "
                + (f"**⚠ {plural(flagged, 'item')} look{'s' if flagged == 1 else ''} like duplicates** (marked below)."
                   if flagged else "No likely duplicates.")
                + (f" **{len(check['errors'])} couldn't be checked:**" if check["errors"] else "")
            )
            for error in check["errors"]:
                st.caption(f"Couldn't check {error}")
        else:
            st.caption(f"To check: {describe_check_scope(entries)} on the Waiting and Withdrawn tabs. "
                       "Each item also links to its record in Salesforce.")


# Actions log: which notes each Agent has actioned, for tracking. A note is logged when its package is sent
# (That's right), marked done when the connector's receipt says its items went through, and logged as done
# when the Agent's mark (e.g. [SF_Tasks]) is found in the note, for things done by hand in Apple Notes.
ACTIONS_LOG_FORMAT = "llm-notes-actions-log/v1"
ACTION_STATUS_BADGES = {"done": "✓", "sent": "⏳", "failed": "⚠", "withdrawn": "↩"}
ACTION_STATUS_TEXT = {"done": "done", "sent": "sent, waiting for the connector", "failed": "failed",
                      "withdrawn": "withdrawn"}
ACTIONED_STATUSES = ("sent", "done")  # what counts as actioned for the badges' tally and the Actioned filter
ACTIONED_FILTER_SETTING = "actioned_filter"  # {"mode": one of ACTIONED_FILTER_MODES, "agent": prompt id or ""}
ACTIONED_FILTER_MODES = {"all": "All notes", "hide": "Not yet actioned", "only": "Only actioned"}
ACTIONS_LOG_RUN_KEY = "actions_log_this_run"  # the log as synced at the top of this run


def actions_mark(prompt: dict) -> str:
    """The text that marks a note as actioned by this Agent: its own, or [<Agent name>]."""
    return str(prompt.get("mark") or "").strip() or f"[{prompt['name']}]"


def agent_prompts(library: dict | None = None) -> list[dict]:
    return [p for p in (library or load_prompt_library())["prompts"] if is_agent_prompt(p)]


def load_actions_log() -> dict:
    log = read_json_file(ACTIONS_LOG_PATH)
    log["format"] = ACTIONS_LOG_FORMAT
    log["entries"] = [e for e in log.get("entries") or [] if isinstance(e, dict)]
    log["marks_scanned"] = log.get("marks_scanned") if isinstance(log.get("marks_scanned"), dict) else {}
    return log


def save_actions_log(log: dict) -> None:
    write_text_atomically(ACTIONS_LOG_PATH, json.dumps(log, indent=2, ensure_ascii=False))
    st.session_state[ACTIONS_LOG_RUN_KEY] = log


def current_actions_log() -> dict:
    """The log as synced this run (see sync_actions_log), or as saved."""
    return st.session_state.get(ACTIONS_LOG_RUN_KEY) or load_actions_log()


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def note_library_key(result: dict, search_state: dict) -> str:
    """The library a result's note is in: all-library and in-scope hits carry their own."""
    return result.get("_safe_lib") or search_state["library"]


def log_sent_actions(search_state: dict, prompts: list[dict], staged: dict, package_file: Path) -> None:
    """Log each note in use against each Agent that staged items from it, as sent."""
    notes = list(dict.fromkeys(
        (note_library_key(r, search_state), str(r.get("file_source", "")))
        for r in discussion_results(search_state) if r.get("file_source")
    ))
    notes = [(lib, note) for lib, note in notes if lib != ALL_LIBRARIES_KEY]
    log = load_actions_log()
    for prompt in prompts:
        indices = [i for i, item in enumerate(staged["items"]) if item.get("_agent_id") == prompt["id"]]
        if not indices and len(prompts) == 1:
            indices = list(range(len(staged["items"])))
        if not indices:
            continue  # this Agent staged nothing, so it actioned nothing
        log["entries"] += [{
            "id": uuid.uuid4().hex, "library": lib, "note": note, "agent_id": prompt["id"], "agent": prompt["name"],
            "mark": actions_mark(prompt), "how": "sent", "status": "sent", "at": now_iso(),
            "package": str(package_file), "items": indices, "links": [],
        } for lib, note in notes]
    save_actions_log(log)


def package_action_status(entry: dict) -> tuple[str, list[str]]:
    """(status, Salesforce links) of a sent entry, from where its package is and the connector's receipt."""
    package_file = Path(entry["package"])
    receipt = read_handoff_receipt(package_file)
    if receipt:
        results = [r for r in receipt.get("items") or [] if r.get("index") in entry.get("items", [])]
        done = [r for r in results if r.get("status") in DONE_ITEM_STATUSES]
        links = [str(r["url"]) for r in done if str(r.get("url") or "").startswith("https://")]
        if done or (not results and receipt.get("status") == "done"):
            return "done", links
        return "failed", []
    if (package_file.parent / WITHDRAWN_DIR / package_file.name).is_file() or not package_file.is_file():
        return "withdrawn", []  # withdrawn on the Actions page, or deleted
    return "sent", []


def refresh_action_statuses(log: dict) -> bool:
    """Bring sent entries up to date with their packages; done is final. Returns True if any changed."""
    changed = False
    for entry in log["entries"]:
        if entry.get("how") != "sent" or entry.get("status") == "done":
            continue
        status, links = package_action_status(entry)
        if status != entry.get("status"):
            entry.update(status=status, links=links)
            if status == "done":
                entry["done_at"] = now_iso()
            changed = True
    return changed


def mark_variants(mark: str) -> list[str]:
    """The mark as typed and as a Markdown export may escape it, e.g. \\[SF\\_Tasks\\]."""
    brackets = re.sub(r"([\[\]])", r"\\\1", mark)
    escaped = re.sub(r"([\\`*_{}\[\]()#+\-.!|])", r"\\\1", mark)
    return list(dict.fromkeys(v.lower() for v in (mark, brackets, escaped)))


def notes_containing(lib_key: str, needles: list[str]) -> set[str]:
    """Notes in a library's text index containing any of the (lower-case) needles."""
    table = lib_key.replace('"', '""')
    clause = " OR ".join("instr(lower(coalesce(text_block, '')), ?) > 0" for _ in needles)
    try:
        with sqlite3.connect(SQLiteConfig.get_uri_string()) as conn:
            return {str(row[0]) for row in conn.execute(
                f'SELECT DISTINCT file_source FROM "{table}" WHERE {clause}', needles)}
    except sqlite3.Error:
        return set()


def scan_action_marks(log: dict, lib_keys: list[str], force: bool = False) -> bool:
    """Log notes holding an Agent's mark as done by hand. A library is scanned when it was rescanned since its
    last scan, the marks changed, or `force`. Entries stay when a mark is later removed. True if any changed."""
    agents = agent_prompts()
    signature = json.dumps(sorted((p["id"], actions_mark(p)) for p in agents))
    rescans = load_text_rescan_status().get("libraries", {})
    logged = {(e["library"], e["note"], e["agent_id"]) for e in log["entries"] if e.get("how") == "mark"}
    changed = False
    for lib in lib_keys:
        scanned = log["marks_scanned"].get(lib) or {}
        rescanned = (rescans.get(lib) or {}).get("at", "")
        stale = (not scanned or scanned.get("marks") != signature
                 or (rescanned and rescanned > scanned.get("rescan", "")))
        if not (force or stale):
            continue
        for prompt in agents:
            for note in sorted(notes_containing(lib, mark_variants(actions_mark(prompt)))):
                if (lib, note, prompt["id"]) not in logged:
                    log["entries"].append({
                        "id": uuid.uuid4().hex, "library": lib, "note": note, "agent_id": prompt["id"],
                        "agent": prompt["name"], "mark": actions_mark(prompt), "how": "mark", "status": "done",
                        "at": now_iso(), "links": [],
                    })
                    logged.add((lib, note, prompt["id"]))
        log["marks_scanned"][lib] = {"at": now_iso(), "rescan": rescanned, "marks": signature}
        changed = True
    return changed


def sync_actions_log(lib_keys: list[str] | None = None, force_marks: bool = False) -> dict:
    """Refresh sent entries and scan for marks in the given libraries (default: every visible corpus; hidden
    ones are skipped, as in rescans), save if anything changed, and keep the result for this run."""
    log = load_actions_log()
    libraries = lib_keys if lib_keys is not None else [key for _name, key, _folder in resolve_text_rescan_targets(True)]
    changed = refresh_action_statuses(log)
    changed = scan_action_marks(log, libraries, force_marks) or changed
    if changed:
        save_actions_log(log)
    st.session_state[ACTIONS_LOG_RUN_KEY] = log
    return log


def note_actions(lib_key: str, note: str) -> list[dict]:
    """The most telling entry per Agent for a note: done before sent, then failed, then withdrawn."""
    order = list(ACTION_STATUS_BADGES)
    best: dict[str, dict] = {}
    for entry in current_actions_log()["entries"]:
        if entry["library"] != lib_key or entry["note"] != note:
            continue
        held = best.get(entry["agent_id"])
        if held is None or (order.index(entry["status"]), entry.get("at", "")) < (order.index(held["status"]), held.get("at", "")):
            best[entry["agent_id"]] = entry
    return sorted(best.values(), key=lambda e: order.index(e["status"]))


def agent_display_name(entry: dict, agents: dict[str, dict]) -> str:
    prompt = agents.get(entry["agent_id"])
    return prompt["name"] if prompt else entry.get("agent", "Agent")


def action_badges(lib_key: str, note: str) -> str:
    """' · ✓ SF_Tasks · ⏳ SF_Next_Step' for a note's label, or ''."""
    agents = {p["id"]: p for p in load_prompt_library()["prompts"]}
    return "".join(f" · {ACTION_STATUS_BADGES[e['status']]} {agent_display_name(e, agents)}"
                   for e in note_actions(lib_key, note))


def render_note_actions(lib_key: str, note: str) -> None:
    """One caption line per Agent that actioned the note: how, when, and links to the Salesforce records."""
    agents = {p["id"]: p for p in load_prompt_library()["prompts"]}
    for entry in note_actions(lib_key, note):
        when = format_when(datetime.datetime.fromisoformat(entry.get("done_at") or entry["at"]))
        how = f"found `{entry['mark']}` in the note" if entry["how"] == "mark" else ACTION_STATUS_TEXT[entry["status"]]
        links = " · ".join(f"[Open record ↗]({url})" for url in entry.get("links", []))
        st.caption(f"{ACTION_STATUS_BADGES[entry['status']]} **{agent_display_name(entry, agents)}**: {how}, "
                   f"{when}" + (f" · {links}" if links else ""))


def actioned_filter_setting() -> dict:
    setting = load_app_settings().get(ACTIONED_FILTER_SETTING) or {}
    mode = setting.get("mode") if setting.get("mode") in ACTIONED_FILTER_MODES else "all"
    return {"mode": mode, "agent": str(setting.get("agent") or "")}


def persist_actioned_filter() -> None:
    save_app_settings({ACTIONED_FILTER_SETTING: {
        "mode": st.session_state.get("actioned_filter_mode") or "all",
        "agent": st.session_state.get("actioned_filter_agent") or "",
    }})


def actioned_sources(lib_key: str, agent_id: str = "") -> set[str]:
    return {e["note"] for e in current_actions_log()["entries"]
            if e["library"] == lib_key and e["status"] in ACTIONED_STATUSES and (not agent_id or e["agent_id"] == agent_id)}


def with_actioned_filter(search_filter: dict, lib_key: str) -> dict:
    """Apply the Actioned filter: leave actioned notes out, or keep only them. Applying it twice changes nothing."""
    setting = actioned_filter_setting()
    if setting["mode"] == "all":
        return search_filter
    sources = actioned_sources(lib_key, setting["agent"])
    if setting["mode"] == "hide":
        return {**search_filter, "exclude_sources": sorted(set(search_filter.get("exclude_sources") or []) | sources)}
    included = search_filter.get("include_sources")
    return {**search_filter, "include_sources": sorted(sources if included is None else sources & set(included))}


def describe_actioned_filter() -> str:
    """'not yet actioned by SF_Tasks', or '' when the filter is off."""
    setting = actioned_filter_setting()
    if setting["mode"] == "all":
        return ""
    prompt = find_prompt(load_prompt_library(), setting["agent"]) if setting["agent"] else None
    by = f"by {prompt['name']}" if prompt else "by any Agent"
    return f"{'not yet actioned' if setting['mode'] == 'hide' else 'actioned'} {by}"


def render_actioned_filter() -> None:
    """Actioned: All notes / Not yet actioned / Only actioned, by one Agent or any."""
    setting = actioned_filter_setting()
    agents = {p["id"]: p["name"] for p in agent_prompts()}
    st.session_state.setdefault("actioned_filter_mode", setting["mode"])
    st.session_state.setdefault("actioned_filter_agent", setting["agent"] if setting["agent"] in agents else "")
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key="actioned_filter_row"):
        st.markdown("Actioned", width="content", help=(
            "Leave out notes an Agent has already actioned, or show only those. A note counts once its package "
            "is sent (⏳) or done (✓), or when the Agent's mark, e.g. `[SF_Tasks]`, is in the note. Failed and "
            "withdrawn packages don't count. Each Agent's mark is set on its card in the Prompt library."))
        st.selectbox("Actioned", options=list(ACTIONED_FILTER_MODES), format_func=ACTIONED_FILTER_MODES.get,
                     key="actioned_filter_mode", on_change=persist_actioned_filter, label_visibility="collapsed",
                     width=170)
        st.selectbox("Agent", options=["", *agents], format_func=lambda key: agents.get(key, "by any Agent"),
                     key="actioned_filter_agent", on_change=persist_actioned_filter, label_visibility="collapsed",
                     width=170, disabled=st.session_state["actioned_filter_mode"] == "all")


def actions_log_backup() -> dict:
    return {"format": ACTIONS_LOG_FORMAT, "exported_at": now_iso(), "entries": load_actions_log()["entries"]}


def parse_actions_log_backup(raw: bytes) -> tuple[list[dict] | None, str]:
    """The entries of an actions log backup, checked, or (None, why not)."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None, "That isn't an actions log backup (not valid JSON)."
    if not isinstance(data, dict) or data.get("format") != ACTIONS_LOG_FORMAT:
        return None, "That isn't an actions log backup from this app."
    entries = data.get("entries")
    required = ("id", "library", "note", "agent_id", "how", "status", "at")
    if not isinstance(entries, list) or not all(
        isinstance(e, dict) and all(e.get(k) for k in required) and e["status"] in ACTION_STATUS_BADGES
        for e in entries
    ):
        return None, "The backup's entries are damaged."
    return entries, ""


def restore_actions_log(entries: list[dict], replace: bool) -> Path:
    """Merge the entries into the log (by id, one mark entry per note and Agent), or replace it. The log as it
    was is saved to the backups folder first; marks are scanned again on the next run. Returns that backup."""
    SETTINGS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = SETTINGS_BACKUP_DIR / f"actions-log-before-restore-{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
    backup.write_text(json.dumps(actions_log_backup(), indent=2, ensure_ascii=False))
    log = load_actions_log()
    kept = [] if replace else log["entries"]
    ids = {e["id"] for e in kept}
    marks = {(e["library"], e["note"], e["agent_id"]) for e in kept if e["how"] == "mark"}
    for entry in entries:
        key = (entry["library"], entry["note"], entry["agent_id"])
        if entry["id"] in ids or (entry["how"] == "mark" and key in marks):
            continue
        kept.append(entry)
        ids.add(entry["id"])
        if entry["how"] == "mark":
            marks.add(key)
    save_actions_log({**log, "entries": kept, "marks_scanned": {}})
    return backup


def restore_actions_log_from_upload() -> None:
    upload = st.session_state.get("actions_log_upload")
    if upload is None:
        return
    entries, problem = parse_actions_log_backup(upload.getvalue())
    if entries is None:
        st.session_state["actions_log_notice"] = ("error", problem)
        return
    replace = st.session_state.get("actions_log_restore_mode") == "Replace"
    backup = restore_actions_log(entries, replace)
    st.session_state["actions_log_notice"] = (
        "success", f"{'Replaced the log with' if replace else 'Merged'} {plural(len(entries), 'entry')}"
        f"{'' if replace else ' (ones already in the log are skipped)'}. The log as it was is in `{backup}`.")


def render_actions_log_backup() -> None:
    """Advanced: back up and restore the actions log."""
    count = len(load_actions_log()["entries"])
    st.caption(f"{plural(count, 'entry')} in `{ACTIONS_LOG_PATH}`. The log isn't in the Configure page's "
               "settings export (it names notes), so back it up here.")
    st.download_button(
        "Back up actions log", data=json.dumps(actions_log_backup(), indent=2, ensure_ascii=False),
        file_name=f"actions-log-{datetime.date.today()}.json", mime="application/json",
        icon=":material/download:", key="actions_log_download", use_container_width=True, disabled=not count,
    )
    st.file_uploader("Restore from a backup", type=["json"], key="actions_log_upload")
    st.radio("Restore", ["Merge", "Replace"], key="actions_log_restore_mode", horizontal=True,
             label_visibility="collapsed",
             help="**Merge** adds the backup's entries you don't have. **Replace** swaps the log for the backup. "
             "Either way, the log as it was is saved to the backups folder first.")
    st.button("Restore actions log", icon=":material/upload:", key="actions_log_restore",
              on_click=restore_actions_log_from_upload, use_container_width=True,
              disabled=st.session_state.get("actions_log_upload") is None)
    if notice := st.session_state.pop("actions_log_notice", None):
        (st.success if notice[0] == "success" else st.error)(notice[1])


def delete_action_entries(entry_ids: list[str]) -> None:
    log = load_actions_log()
    log["entries"] = [e for e in log["entries"] if e["id"] not in set(entry_ids)]
    save_actions_log(log)
    st.session_state["actions_notice"] = f"Deleted {plural(len(entry_ids), 'entry')} from the actions log."


def scan_marks_now() -> None:
    sync_actions_log(force_marks=True)
    st.session_state["actions_notice"] = "Scanned the indexed notes for Agents' marks."


def render_actions_log() -> None:
    """The Actions page's log tab: every note an Agent actioned, with filters and delete."""
    log = sync_actions_log()
    agents = {p["id"]: p for p in load_prompt_library()["prompts"]}
    marks = ", ".join(f"`{actions_mark(p)}`" for p in agent_prompts())
    st.caption(
        "Notes logged when an Agent's package is sent (⏳), done once the connector's receipt says its items went "
        f"through (✓), or failed (⚠) or withdrawn (↩). Notes holding an Agent's mark ({marks or 'none yet'}) are "
        "logged as done by hand, after a rescan finds the mark. Entries stay if a mark is later removed."
    )
    entries = log["entries"]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        agent_ids = list(dict.fromkeys(e["agent_id"] for e in entries))
        chosen_agents = st.multiselect("Agent", agent_ids, key="actions_log_agents",
                                       format_func=lambda key: agents[key]["name"] if key in agents else next(
                                           (e.get("agent", key) for e in entries if e["agent_id"] == key), key),
                                       placeholder="All Agents")
        chosen_status = st.multiselect("Status", list(ACTION_STATUS_BADGES), key="actions_log_status",
                                       format_func=lambda s: f"{ACTION_STATUS_BADGES[s]} {s}", placeholder="Any status")
        query = st.text_input("Note", key="actions_log_query", placeholder="Search note names")
        st.button("Scan notes for marks now", icon=":material/sell:", key="actions_log_scan", on_click=scan_marks_now,
                  help="Look for each Agent's mark in every indexed note now, rather than after the next rescan.")
    shown = sorted(
        (e for e in entries if (not chosen_agents or e["agent_id"] in chosen_agents)
         and (not chosen_status or e["status"] in chosen_status)
         and query.strip().lower() in e["note"].lower()),
        key=lambda e: e.get("done_at") or e["at"], reverse=True,
    )
    if not entries:
        st.caption("Nothing logged yet.")
        return
    st.caption(f"{len(shown)} of {plural(len(entries), 'entry')}.")
    rows = [{
        "Status": f"{ACTION_STATUS_BADGES[e['status']]} {e['status']}",
        "Note": Path(e["note"]).name,
        "Library": e["library"],
        "Agent": agent_display_name(e, agents),
        "How": f"mark {e['mark']}" if e["how"] == "mark" else "sent",
        "When": format_when(datetime.datetime.fromisoformat(e.get("done_at") or e["at"])),
        "Record": (e.get("links") or [None])[0],
    } for e in shown]
    selection = st.dataframe(
        rows, key="actions_log_table", on_select="rerun", selection_mode="multi-row", hide_index=True,
        use_container_width=True,
        column_config={"Record": st.column_config.LinkColumn("Record", display_text="Open ↗")},
    )
    picked = [shown[i]["id"] for i in selection.selection.rows if i < len(shown)]
    with st.popover(f"Delete {len(picked)} selected" if picked else "Delete selected", icon=":material/delete:",
                    disabled=not picked, help="Select rows in the table to delete them from the log"):
        st.markdown(f"Delete {plural(len(picked), 'entry')} from the actions log? A note still holding its mark is "
                    "logged again on the next scan.")
        st.button("Delete", key="actions_log_delete", type="primary", on_click=delete_action_entries, args=(picked,))


def render_actions_page() -> None:
    """Actions page (?page=actions): everything Agents sent to the outbox and what the connector did with it."""
    render_outbox_status(link=False)
    render_page_title("Actions")
    outbox = get_outbox_dir()
    # The task executor (defined in Task executor settings on the Configure page) processes what's listed here.
    executor = executor_link(load_prompt_library())
    if executor and executor.get("url"):
        st.markdown(
            f"**Task executor:** {link_markdown(executor)} — processes the outbox; open it to run now or change "
            "the schedule."
        )
    else:
        st.caption(f"Set the **Task executor** in **Task executor settings** on the [Configure]({CONFIGURE_PAGE_URL}) "
                   "page to open it from here.")
    st.caption(
        f"Everything Agents have sent to the outbox (`{outbox}`) and what the connector did with it. "
        "Edit a package's text before a connector acts on it, withdraw it to stop a connector acting on it, "
        "or retry one after a temporary error."
    )
    if notice := st.session_state.pop("actions_notice", None):
        st.toast(notice)
    if offered := st.session_state.pop(OFFER_FIX_KEY, None):
        offer_semantic_fix_dialog(offered)
    entries = load_outbox_entries(outbox)
    # "View in outbox" in the Agent box links here with ?package=<file name>: start filtered to it.
    if (linked := st.query_params.get("package")) and "actions_query" not in st.session_state:
        st.session_state["actions_query"] = Path(linked).stem
    with st.container(horizontal=True, vertical_alignment="bottom"):
        with st.container(horizontal=True, vertical_alignment="bottom", gap="xsmall"):
            query = st.text_input(
                "Filter",
                key="actions_query",
                placeholder="Search subjects, titles, prompts, owners and messages",
                icon=":material/search:",
            )
            st.button(
                ":material/close:",
                key="actions_clear_query",
                type="tertiary",
                on_click=st.session_state.update,
                kwargs={"actions_query": ""},
                disabled=not query,
                help="Clear the filter",
            )
        st.button("Refresh", icon=":material/refresh:", key="actions_refresh")
    render_duplicate_check_panel(entries)
    check = st.session_state.get(DUPLICATE_CHECK_KEY) or {}
    flagged = {state: sum(1 for i in outbox_items({state: entries[state]})
                          if any(m["text"] == i["text"] for m in check.get("matches", {}).get(i["key"], [])))
               for state in DUPLICATE_CHECK_STATES}
    # Tabs count packages and the items in them, as the duplicate check counts items.
    item_counts = {state: len(outbox_items({state: entries[state]})) for state in OUTBOX_STATES}
    log_notes = {(e["library"], e["note"]) for e in sync_actions_log()["entries"]}
    tabs = st.tabs([
        f"{OUTBOX_STATES[state][0]} ({plural(len(entries[state]), 'package')}"
        + (f" · {plural(item_counts[state], 'item')}" if entries[state] else "") + ")"
        + (f" · ⚠ {flagged[state]}" if flagged.get(state) else "") for state in OUTBOX_STATES
    ] + [f"Actions log ({plural(len(log_notes), 'note')})"])
    with tabs[-1]:
        render_actions_log()
    for tab, state in zip(tabs, OUTBOX_STATES):
        with tab:
            shown = [entry for entry in entries[state] if outbox_entry_matches(entry, query)]
            if query.strip() and entries[state]:
                st.caption(f"{len(shown)} of {len(entries[state])} match.")
            if not shown:
                st.caption(OUTBOX_STATES[state][1] if not entries[state] else "Nothing matches the filter.")
            if state == "withdrawn" and shown:
                everything = len(shown) == len(entries[state])
                label = "Delete all" if everything else f"Delete these {len(shown)}"
                with st.popover(label, icon=":material/delete_sweep:",
                                help="Delete the withdrawn packages listed here for good"):
                    st.markdown(f"Delete {'all ' if everything else ''}{len(shown)} withdrawn package"
                                f"{'s' if len(shown) != 1 else ''} and their receipts for good? This can't be undone.")
                    st.button(
                        label,
                        key="actions_delete_withdrawn",
                        type="primary",
                        on_click=delete_withdrawn_packages,
                        args=([str(entry["path"]) for entry in shown],),
                    )
            for entry in shown:
                render_outbox_entry(entry, state)


PROMPT_LIBRARY_PAGE_URL = "?page=prompts"
CONFIGURE_PAGE_URL = "?page=configure"


def save_title_marker() -> None:
    library = load_prompt_library()
    current = get_title_marker(library)
    position = st.session_state.get("title_marker_position") or current["position"]  # deselecting keeps it
    library["title_marker"] = {
        "text": str(st.session_state.get("title_marker_text", current["text"])).strip(),
        "position": position,
    }
    save_prompt_library(library)


def save_profile_from_widgets() -> None:
    library = load_prompt_library()
    profile = library["profile"]
    library["profile"] = {
        **profile,  # keep fields whose widgets aren't on screen
        "text": str(st.session_state.get("profile_text", profile.get("text", ""))),
        "enabled": bool(st.session_state.get("profile_enabled", profile.get("enabled", True))),
        "short_name": re.sub(r"\s+", "", str(st.session_state.get("profile_short_name", profile.get("short_name", "")))),
    }
    save_prompt_library(library)


def save_profile_text() -> None:
    save_profile_from_widgets()
    mark_text_saved("profile_text")


def save_prompt_field(prompt_id: str, field: str) -> None:
    value = st.session_state.get(f"prompt_{field}_{prompt_id}")
    if field == "tags":
        value = clean_filter_terms(value)
    elif field in ("global", "starred", "agent", "salesforce"):
        value = bool(value)
    elif field == "output":
        value = value if value in DRAFT_OUTPUT_TYPES else DEFAULT_AGENT_OUTPUT
    elif field == "model":
        value = str(value or "")
    elif field in ("links", "references"):
        value = [str(item_id) for item_id in (value or [])]
    elif field == "mark":
        value = str(value or "").strip()  # empty means the default, [<Agent name>]
    else:
        value = str(value or "").strip()
        if not value:
            return  # keep the old name/text/command rather than saving an empty one
    before = find_prompt(load_prompt_library(), prompt_id) or {}
    update_prompt(prompt_id, **{field: value})
    if field == "agent" and value:
        output = before.get("output") if before.get("output") in DRAFT_OUTPUT_TYPES else DEFAULT_AGENT_OUTPUT
        update_prompt(prompt_id, output=output, command=before.get("command") or DEFAULT_AGENT_COMMANDS[output])
    if field == "output" and before.get("command", "") in ("", *DEFAULT_AGENT_COMMANDS.values()):
        update_prompt(prompt_id, command=DEFAULT_AGENT_COMMANDS[value])  # keep a default command in step with the output
    if field in ("name", "text", "command", "mark"):
        mark_text_saved(f"prompt_{prompt_id}")


def update_prompt_steps(prompt_id: str, change) -> None:
    """Apply `change(steps)` to an Agent prompt's middle steps and save."""
    library = load_prompt_library()
    prompt = find_prompt(library, prompt_id)
    if prompt:
        prompt["steps"] = change(list(prompt.get("steps", [])))
        save_prompt_library(library)


def add_prompt_step(prompt_id: str) -> None:
    update_prompt_steps(prompt_id, lambda steps: steps + [{"id": uuid.uuid4().hex[:8], "name": "New step", "text": ""}])


def delete_prompt_step(prompt_id: str, step_id: str) -> None:
    update_prompt_steps(prompt_id, lambda steps: [s for s in steps if s["id"] != step_id])


def move_prompt_step(prompt_id: str, step_id: str, offset: int) -> None:
    def move(steps: list[dict]) -> list[dict]:
        index = next(i for i, s in enumerate(steps) if s["id"] == step_id)
        target = min(max(index + offset, 0), len(steps) - 1)
        steps.insert(target, steps.pop(index))
        return steps

    update_prompt_steps(prompt_id, move)


def save_prompt_step(prompt_id: str, step_id: str, field: str) -> None:
    value = str(st.session_state.get(f"prompt_step_{field}_{step_id}", "")).strip()
    if field == "name" and not value:
        return  # a step always keeps a name for its button
    update_prompt_steps(prompt_id, lambda steps: [{**s, field: value} if s["id"] == step_id else s for s in steps])
    mark_text_saved(f"prompt_step_{step_id}")


def add_prompt_from_form() -> None:
    name = str(st.session_state.get("new_prompt_name", "")).strip()
    text = str(st.session_state.get("new_prompt_text", "")).strip()
    if not name or not text:
        st.session_state["add_prompt_error"] = "Give the prompt a name and some text."
        return
    global_prompt = bool(st.session_state.get("new_prompt_global", False))
    agent = bool(st.session_state.get("new_prompt_agent", False))
    if global_prompt and agent:
        st.session_state["add_prompt_error"] = "A prompt can be Global or Agent, not both."
        return
    library = load_prompt_library()
    library["prompts"].append(
        new_prompt_record(
            name,
            text,
            global_prompt=global_prompt,
            tags=st.session_state.get("new_prompt_tags", []),
            starred=bool(st.session_state.get("new_prompt_starred", False)),
            agent=agent,
            output=st.session_state.get("new_prompt_output", DEFAULT_AGENT_OUTPUT) if agent else "text",
            model=st.session_state.get("new_prompt_model") or "",
            references=[ref_id for ref_id in st.session_state.get("new_prompt_references", [])
                        if find_reference_document(ref_id, library)],
            salesforce=bool(st.session_state.get("new_prompt_salesforce", False)),
        )
    )
    save_prompt_library(library)
    st.session_state["add_prompt_added"] = name


def set_prompt_page_query(text: str) -> None:
    st.session_state["prompt_page_query"] = text


def prompt_model_options(current: str = "") -> list[str]:
    """Choices for a prompt's Model box: '' (use the selected model), then the installed Ollama models."""
    ok, installed = discover_ollama_models()
    models = list(installed if ok and installed else SUGGESTED_OLLAMA_MODELS)
    if current and current not in models:
        models.insert(0, current)  # keep a saved model that isn't installed right now
    return ["", *models]


def format_prompt_model(model: str) -> str:
    return model or "Model: as selected"


def show_all_prompts() -> None:
    st.query_params.pop("prompt", None)


def common_prompt_tags(library: dict, limit: int = 8) -> list[str]:
    """Tags of the most recently used prompts first (then newest prompts), without duplicates."""
    ordered = sorted(
        library["prompts"],
        key=lambda p: (p.get("last_used") or "", p.get("created") or ""),
        reverse=True,
    )
    tags: list[str] = []
    for prompt in ordered:
        for tag in prompt.get("tags", []):
            if tag.lower() not in (t.lower() for t in tags):
                tags.append(tag)
    return tags[:limit]


def render_agent_workflow_editor(prompt: dict) -> None:
    """Middle steps and the Action Command of an Agent prompt; step 1 is the prompt text above."""
    prompt_id = prompt["id"]
    steps = prompt.get("steps", [])
    for number, step in enumerate(steps, 2):
        step_id = step["id"]
        with st.container(horizontal=True, vertical_alignment="center", gap="xsmall"):
            st.caption(f"**Step {number}**", width="content")
            st.text_input(
                "Step name",
                value=step["name"],
                key=f"prompt_step_name_{step_id}",
                on_change=save_prompt_step,
                args=(prompt_id, step_id, "name"),
                label_visibility="collapsed",
                help="Shown on the step's button in the Discussion panel.",
            )
            st.button(":material/arrow_upward:", key=f"prompt_step_up_{step_id}", type="tertiary",
                      disabled=number == 2, on_click=move_prompt_step, args=(prompt_id, step_id, -1), help="Move up")
            st.button(":material/arrow_downward:", key=f"prompt_step_down_{step_id}", type="tertiary",
                      disabled=number == len(steps) + 1, on_click=move_prompt_step, args=(prompt_id, step_id, 1),
                      help="Move down")
            st.button(":material/close:", key=f"prompt_step_delete_{step_id}", type="tertiary",
                      on_click=delete_prompt_step, args=(prompt_id, step_id), help="Remove this step")
        st.text_area(
            "Step prompt",
            value=step["text"],
            key=f"prompt_step_text_{step_id}",
            on_change=save_prompt_step,
            args=(prompt_id, step_id, "text"),
            height=68,
            placeholder="e.g. Ask me for a due date and an owner for each task I selected.",
            label_visibility="collapsed",
        )
        render_saved_note(f"prompt_step_{step_id}")
    st.button(
        "Add step",
        icon=":material/add:",
        key=f"prompt_step_add_{prompt_id}",
        on_click=add_prompt_step,
        args=(prompt_id,),
        help="A step is a prompt you can send with one click while refining, e.g. 'Select tasks' or 'Set due dates'.",
    )
    st.text_input(
        "Action Command",
        value=prompt.get("command", ""),
        key=f"prompt_command_{prompt_id}",
        on_change=save_prompt_field,
        args=(prompt_id, "command"),
        placeholder="e.g. Send to tasks in Salesforce",
        help=(
            "The last step. Its button in the Discussion panel stages what you agreed as "
            f"{OUTPUT_TYPES[get_prompt_output_type(prompt)].lower()}; you confirm with **That's right** "
            "before anything goes to the outbox."
        ),
    )
    st.text_input(
        "Actioned mark",
        value=prompt.get("mark", ""),
        key=f"prompt_mark_{prompt_id}",
        on_change=save_prompt_field,
        args=(prompt_id, "mark"),
        placeholder=f"[{prompt['name']}]",
        help=(
            "Tracks which notes this Agent has actioned. Notes it sends are logged in the **Actions log** "
            "automatically. Type this mark into a note in Apple Notes to log something you did by hand: the next "
            f"rescan finds it. Leave empty to use `[{prompt['name']}]`."
        ),
    )


OPEN_PROMPTS_SETTING = "prompt_cards_open"  # ids of the prompt cards left open on the prompt library page


def saved_open_prompts() -> set[str]:
    return set(load_app_settings().get(OPEN_PROMPTS_SETTING) or [])


def persist_open_prompt(prompt_id: str) -> None:
    """Remember whether a prompt card is open (saved with the app's settings)."""
    opened = saved_open_prompts()
    (opened.add if st.session_state.get(f"prompt_open_{prompt_id}") else opened.discard)(prompt_id)
    save_app_settings({OPEN_PROMPTS_SETTING: sorted(opened)})


def set_prompts_open(prompt_ids: list[str], is_open: bool) -> None:
    """Expand all / Collapse all for one section of the prompt library."""
    for prompt_id in prompt_ids:
        st.session_state[f"prompt_open_{prompt_id}"] = is_open
    opened = saved_open_prompts()
    opened = opened | set(prompt_ids) if is_open else opened - set(prompt_ids)
    save_app_settings({OPEN_PROMPTS_SETTING: sorted(opened)})


def prompt_card_label(prompt: dict) -> str:
    """A collapsed card's header: the name as the Ask box shows it, plus Global and Starred markers."""
    markers = (" · Global" if prompt.get("global") else "") + (" · ★" if prompt.get("starred") else "")
    return agent_prompt_label(prompt) + markers


def render_prompt_card(
    prompt: dict, model_options: list[str], links: list[dict] = (), references: list[dict] = ()
) -> None:
    prompt_id = prompt["id"]
    agent = is_agent_prompt(prompt)
    # The key gives the card a CSS class; Agent cards get a blue border (see render_prompt_library_page).
    # Collapsible: a header row (the name and its markers) opens the card; open cards are remembered.
    open_key = f"prompt_open_{prompt_id}"
    if open_key not in st.session_state:
        st.session_state[open_key] = prompt_id in saved_open_prompts()
    with st.container(key=f"{'agent' if agent else 'task'}_prompt_card_{prompt_id}"):
        with st.expander(prompt_card_label(prompt), key=open_key, on_change=persist_open_prompt, args=(prompt_id,)):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.text_input(
                    "Name",
                    value=prompt["name"],
                    key=f"prompt_name_{prompt_id}",
                    on_change=save_prompt_field,
                    args=(prompt_id, "name"),
                    label_visibility="collapsed",
                )
                with st.popover(":material/delete:", type="tertiary", help="Delete this prompt"):
                    st.markdown(f"Delete **{prompt['name']}**? This can't be undone.")
                    st.button(
                        "Delete",
                        key=f"prompt_delete_{prompt_id}",
                        type="primary",
                        icon=":material/delete:",
                        on_click=delete_prompt,
                        args=(prompt_id,),
                    )
            if agent:
                st.caption("**Step 1 · Collection** — what to find in the notes, list and ask about.")
            st.text_area(
                "Prompt",
                value=prompt["text"],
                key=f"prompt_text_{prompt_id}",
                on_change=save_prompt_field,
                args=(prompt_id, "text"),
                height=90,
                label_visibility="collapsed",
            )
            render_saved_note(f"prompt_{prompt_id}")
            with st.container(horizontal=True, vertical_alignment="center"):
                st.checkbox(
                    "Global",
                    value=bool(prompt.get("global")),
                    key=f"prompt_global_{prompt_id}",
                    on_change=save_prompt_field,
                    args=(prompt_id, "global"),
                    disabled=agent,
                    help=(
                        "Send this with **every** LLM request in **every** corpus, like About me — for standing "
                        "instructions. It can still be left out of a single run in the Ask panel. "
                        "Not available for Agent prompts."
                    ),
                )
                st.checkbox(
                    "Starred",
                    value=bool(prompt.get("starred")),
                    key=f"prompt_starred_{prompt_id}",
                    on_change=save_prompt_field,
                    args=(prompt_id, "starred"),
                    help="Starred prompts fill the Ask panel's Starred row (Agents: its Agent Starred row).",
                )
                st.checkbox(
                    "Agent",
                    value=agent,
                    key=f"prompt_agent_{prompt_id}",
                    on_change=save_prompt_field,
                    args=(prompt_id, "agent"),
                    disabled=bool(prompt.get("global")),
                    help=(
                        "Agent prompts are a workflow: step 1 collects and lists what's in the notes, optional steps "
                        "refine it with you, and the **Action Command** stages a **Note**, **Tasks** or an **Email** "
                        "for you to confirm and send to the outbox for another system (e.g. Salesforce). Only Agent "
                        "prompts send to the outbox. Not available for Global prompts."
                    ),
                )
                if agent:
                    st.selectbox(
                        "Output",
                        options=DRAFT_OUTPUT_TYPES,
                        index=DRAFT_OUTPUT_TYPES.index(get_prompt_output_type(prompt)),
                        format_func=OUTPUT_TYPES.get,
                        key=f"prompt_output_{prompt_id}",
                        on_change=save_prompt_field,
                        args=(prompt_id, "output"),
                        width=150,
                        label_visibility="collapsed",
                        help="What this Agent drafts for the outbox.",
                    )
                options = model_options if prompt.get("model", "") in model_options else [*model_options, prompt["model"]]
                st.selectbox(
                    "Model",
                    options=options,
                    index=options.index(prompt.get("model", "")),
                    format_func=format_prompt_model,
                    key=f"prompt_model_{prompt_id}",
                    on_change=save_prompt_field,
                    args=(prompt_id, "model"),
                    width=200,
                    label_visibility="collapsed",
                    help=(
                        "Model to use when you pick this prompt, for the answer and the discussion. You can still "
                        "change either model in the Ask panel. **Model: as selected** keeps whatever is selected."
                    ),
                )
                st.multiselect(
                    "Tags",
                    options=prompt.get("tags", []),
                    default=prompt.get("tags", []),
                    key=f"prompt_tags_{prompt_id}",
                    accept_new_options=True,
                    on_change=save_prompt_field,
                    args=(prompt_id, "tags"),
                    placeholder="Add a tag, press Enter",
                    label_visibility="collapsed",
                )
            if links:
                titles = {
                    link["id"]: ("⚙️ " if link.get("role") == EXECUTOR_ROLE else "🔗 ") + (link["title"] or link["url"])
                    for link in links
                }
                st.multiselect(
                    "Links",
                    options=list(titles),
                    default=[link_id for link_id in prompt.get("links", []) if link_id in titles],
                    format_func=lambda link_id: titles.get(link_id, link_id),
                    key=f"prompt_links_{prompt_id}",
                    on_change=save_prompt_field,
                    args=(prompt_id, "links"),
                    placeholder="Links to show when this prompt is used",
                    label_visibility="collapsed",
                    help="Shown as links under the question in the Ask box while this prompt is injected. "
                    "Add links in **Task executor settings** on the Configure page.",
                )
            if references:
                ref_titles = {ref["id"]: "📄 " + reference_label(ref) for ref in references}
                st.multiselect(
                    "Reference documents",
                    options=list(ref_titles),
                    default=[ref_id for ref_id in prompt.get("references", []) if ref_id in ref_titles],
                    format_func=lambda ref_id: ref_titles.get(ref_id, ref_id),
                    key=f"prompt_references_{prompt_id}",
                    on_change=save_prompt_field,
                    args=(prompt_id, "references"),
                    placeholder="Reference documents to send with this prompt",
                    label_visibility="collapsed",
                    help="Sent to the LLM as background whenever this prompt is used, Agent or not. "
                    "Make them in **Reference documents** at the top of this page.",
                )
            st.checkbox(
                "☁️ Send Salesforce records",
                value=bool(prompt.get("salesforce")),
                key=f"prompt_salesforce_{prompt_id}",
                on_change=save_prompt_field,
                args=(prompt_id, "salesforce"),
                help=SALESFORCE_PROMPT_HELP,
            )
            if agent:
                render_agent_workflow_editor(prompt)
            if prompt.get("last_used"):
                st.caption(f"Last used {prompt['last_used'].replace('T', ' ')}")


PROMPT_SECTIONS_SETTING = "prompt_sections_open"  # {"task": bool, "agent": bool}; both open by default
PROMPT_SECTION_KEYS = {"task": "task_prompts", "agent": "agent_prompts"}


def persist_prompt_section(section: str) -> None:
    opened = dict(load_app_settings().get(PROMPT_SECTIONS_SETTING) or {})
    opened[section] = bool(st.session_state.get(PROMPT_SECTION_KEYS[section]))
    save_app_settings({PROMPT_SECTIONS_SETTING: opened})


def prompt_section(section: str, title: str, icon: str):
    """The Task prompts or Agent prompts section: collapsible, open until closed, remembered."""
    key = PROMPT_SECTION_KEYS[section]
    if key not in st.session_state:
        st.session_state[key] = (load_app_settings().get(PROMPT_SECTIONS_SETTING) or {}).get(section, True)
    if st.query_params.get("prompt"):  # an Agent's Edit prompt workflow link: show the card it opens
        st.session_state[key] = True
    return st.expander(title, key=key, icon=icon, on_change=persist_prompt_section, args=(section,))


def render_prompt_section_cards(
    section: str, prompts: list[dict], library: dict, model_options: list[str], empty: str
) -> None:
    """One section's prompt cards (collapsed until opened), with Expand all / Collapse all."""
    if not prompts:
        st.caption(empty)
        return
    ids = [p["id"] for p in prompts]
    with st.container(horizontal=True, gap="small"):
        st.button("Expand all", key=f"prompts_expand_{section}", icon=":material/unfold_more:", type="tertiary",
                  on_click=set_prompts_open, args=(ids, True))
        st.button("Collapse all", key=f"prompts_collapse_{section}", icon=":material/unfold_less:", type="tertiary",
                  on_click=set_prompts_open, args=(ids, False))
    for index, prompt in enumerate(prompts):
        # The card, with ↑ / ↓ beside it to move it within its section.
        with st.container(horizontal=True, vertical_alignment="top", gap="xsmall"):
            render_prompt_card(prompt, model_options, ordered_links(library), reference_documents(library))
            with st.container(horizontal=True, gap=None, width="content"):
                st.button(":material/arrow_upward:", key=f"prompt_up_{prompt['id']}", type="tertiary",
                          disabled=index == 0, on_click=swap_prompts, args=(prompt["id"], ids[index - 1]),
                          help="Move up")
                st.button(":material/arrow_downward:", key=f"prompt_down_{prompt['id']}", type="tertiary",
                          disabled=index == len(prompts) - 1, on_click=swap_prompts,
                          args=(prompt["id"], ids[min(index + 1, len(ids) - 1)]), help="Move down")


def swap_prompts(first_id: str, second_id: str) -> None:
    """Swap two prompts' places in the library (its order is the order everywhere prompts are listed)."""
    library = load_prompt_library()
    order = [p["id"] for p in library["prompts"]]
    if first_id in order and second_id in order and first_id != second_id:
        i, j = order.index(first_id), order.index(second_id)
        library["prompts"][i], library["prompts"][j] = library["prompts"][j], library["prompts"][i]
        save_prompt_library(library)


# --- Export and import of all settings (no note content) --------------------------------------------------
# One JSON file with the notes app's settings and prompts, the Salesforce connector's config.toml and the Notes
# exporter's config.json, so that after the notes are put back and indexed everything is as it was. Nothing
# from the notes themselves: no text, no indexes, no references to particular notes.

SETTINGS_FORMAT = "rag-workspace-settings/v1"
NOTE_DERIVED_SETTINGS = {"excluded_notes", "last_question", "search_filter", "llm_speed"}  # about notes, not settings
OTHER_CONFIG_DEFAULTS = {
    "connector": PROJECT_DIR.parent / "rag-action-interface" / "config.toml",  # the Salesforce connector
    "exporter": Path("~/.config/notes-export/config.json").expanduser(),  # the Notes exporter
}
OTHER_CONFIG_LABELS = {"connector": "Salesforce connector", "exporter": "Notes exporter"}
SETTINGS_BACKUP_DIR = APP_DATA_DIR / "backups"


def other_config_path(kind: str) -> Path:
    """Where the connector's or exporter's settings file is (set on the Configure page, else its default)."""
    raw = str(load_app_settings().get(f"{kind}_config_path") or "").strip()
    return Path(raw).expanduser() if raw else OTHER_CONFIG_DEFAULTS[kind]


def persist_other_config_path(kind: str) -> None:
    save_app_settings({f"{kind}_config_path": str(st.session_state.get(f"{kind}_config_path_input", "")).strip()})


def read_json_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def build_settings_bundle(parts: set[str], full: bool = False) -> dict:
    """The settings file. `full` (local backups only) keeps what an export leaves out about notes."""
    bundle = {"format": SETTINGS_FORMAT, "exported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "notes_app": None, "connector": None, "exporter": None}
    if "notes_app" in parts:
        app = read_json_file(APP_SETTINGS_PATH)
        library = json.loads(json.dumps(load_prompt_library()))
        if not full:
            app = {key: value for key, value in app.items() if key not in NOTE_DERIVED_SETTINGS}
            library.pop("references", None)  # reference documents point at notes
            for prompt in library["prompts"]:
                prompt["references"] = []
        bundle["notes_app"] = {"app_settings": app, "prompt_library": library,
                               "corpus_settings": read_json_file(CORPUS_SETTINGS_PATH)}
    connector = other_config_path("connector")
    if "connector" in parts and connector.is_file():
        bundle["connector"] = {"path": str(connector), "config_toml": connector.read_text()}
    exporter = other_config_path("exporter")
    if "exporter" in parts and exporter.is_file():
        config = read_json_file(exporter)
        if not full:
            for key in ("include_notes", "exclude_notes"):  # single notes, named by title or Apple Notes ID
                (config.get("filter") or {}).pop(key, None)
        bundle["exporter"] = {"path": str(exporter), "config": config}
    return bundle


def parse_settings_bundle(raw: bytes) -> tuple[dict | None, str]:
    """A settings file read and checked, or (None, why not)."""
    import tomllib

    try:
        bundle = json.loads(raw)
    except ValueError:
        return None, "That isn't a settings file (not valid JSON)."
    if not isinstance(bundle, dict) or bundle.get("format") != SETTINGS_FORMAT:
        return None, f"That isn't a settings file from this app (format {bundle.get('format') if isinstance(bundle, dict) else '?'!r})."
    notes = bundle.get("notes_app")
    if notes is not None and not (isinstance(notes, dict) and isinstance(notes.get("prompt_library"), dict)
                                  and isinstance(notes.get("app_settings"), dict)):
        return None, "The notes app part of the file is damaged."
    if bundle.get("connector"):
        try:
            tomllib.loads(bundle["connector"].get("config_toml", ""))
        except (tomllib.TOMLDecodeError, AttributeError):
            return None, "The Salesforce connector part of the file is damaged."
    if bundle.get("exporter") and not isinstance(bundle["exporter"].get("config"), dict):
        return None, "The Notes exporter part of the file is damaged."
    return bundle, ""


def write_settings_backup() -> Path:
    """Everything as it is now (including what exports leave out), before an import replaces it."""
    SETTINGS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = SETTINGS_BACKUP_DIR / f"settings-before-import-{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(build_settings_bundle({"notes_app", "connector", "exporter"}, full=True), indent=2))
    return path


def write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text)
    temp.replace(path)


def apply_settings_bundle(bundle: dict, parts: set[str]) -> tuple[list[str], Path]:
    """Restore the chosen parts, to the paths in the file. Returns (what was restored, the backup made first)."""
    backup = write_settings_backup()
    restored = []
    notes = bundle.get("notes_app")
    if "notes_app" in parts and notes:
        app = {**read_json_file(APP_SETTINGS_PATH), **notes["app_settings"]}
        write_text_atomically(APP_SETTINGS_PATH, json.dumps(app, indent=2))
        library = dict(notes["prompt_library"])
        library["references"] = load_prompt_library().get("references", [])  # this machine's own, if any
        save_prompt_library(library)
        corpora = {**read_json_file(CORPUS_SETTINGS_PATH), **(notes.get("corpus_settings") or {})}
        write_text_atomically(CORPUS_SETTINGS_PATH, json.dumps(corpora, indent=2))
        restored.append(f"the notes app ({len(library.get('prompts', []))} prompts, {len(corpora)} corpora)")
    for kind in ("connector", "exporter"):
        part = bundle.get(kind)
        if kind not in parts or not part:
            continue
        path = Path(part["path"]).expanduser()
        text = part["config_toml"] if kind == "connector" else json.dumps(part["config"], indent=2)
        write_text_atomically(path, text)
        save_app_settings({f"{kind}_config_path": str(path)})
        restored.append(f"the {OTHER_CONFIG_LABELS[kind]} (`{path}`)")
    return restored, backup


def describe_settings_bundle(bundle: dict) -> dict[str, str]:
    """{part: one line on what it holds} for the import's checkboxes."""
    lines = {}
    if notes := bundle.get("notes_app"):
        prompts = notes["prompt_library"].get("prompts", [])
        agents = sum(1 for prompt in prompts if prompt.get("agent") and not prompt.get("global"))
        lines["notes_app"] = (
            f"Notes app: {len(prompts)} prompts ({agents} Agents), {len(notes['prompt_library'].get('links', []))} "
            f"links, settings for {len(notes.get('corpus_settings') or {})} corpora; notes folder "
            f"`{notes['app_settings'].get('corpus_root_path', '?')}`"
        )
    if connector := bundle.get("connector"):
        import tomllib

        org = tomllib.loads(connector["config_toml"]).get("org", "?")
        lines["connector"] = f"Salesforce connector: org `{org}`, to `{connector['path']}`"
    if exporter := bundle.get("exporter"):
        lines["exporter"] = (f"Notes exporter: export folder `{exporter['config'].get('output', '?')}`, to "
                             f"`{exporter['path']}`")
    return lines


def index_all_corpora_here() -> None:
    """After an import: index every visible corpus, each with its restored embedding model (Rescan all corpora)."""
    corpus_root = get_corpus_root_path()
    if corpus_root is None or not corpus_root.exists():
        st.error("The notes folder from the settings isn't there yet: put the notes back, then index.")
        return
    if is_single_corpus_mode():
        st.info("Single-corpus mode: open the RAG workspace and click **⚡ Rescan (incremental)** in the left panel.")
        return
    set_embedding_acceleration(get_use_gpu_for_embeddings())
    summary = run_multi_corpus_rescan(filter_libraries_for_display(get_libraries()), corpus_root, st.container())
    st.session_state["multi_rebuild_summary"] = summary
    st.success("Indexing finished. Reload the RAG workspace tab to search.")


def render_settings_transfer_section() -> None:
    """Configure page: export all settings to a file, or import them from one."""
    with st.expander("Export and import settings", expanded=True, key="configure_transfer_open",
                     icon=":material/import_export:"):
        st.caption(
            "Settings only, for moving to another Mac or starting again: prompts (with About me and Agents), "
            "links, quick search terms, the title marker, each corpus's settings and corpus prompt, the app's "
            "settings, and the Salesforce connector's and Notes exporter's settings. **Nothing from your notes:** "
            "no text, no indexes, no reference documents, no excluded notes, filter terms or last question, and "
            "none of the exporter's single-note exceptions. To restore: put the notes back where they were "
            "(e.g. export them again), import this file, then index."
        )
        st.markdown("**Export**")
        found = {}
        for kind, label in OTHER_CONFIG_LABELS.items():
            path = other_config_path(kind)
            found[kind] = path.is_file()
            st.text_input(f"{label} settings file", value=str(path), key=f"{kind}_config_path_input",
                          on_change=persist_other_config_path, args=(kind,),
                          help="✓ found" if found[kind] else "Not found here: it won't be in the export.")
        with st.container(horizontal=True, vertical_alignment="center"):
            parts = {"notes_app"}
            for kind, label in OTHER_CONFIG_LABELS.items():
                if st.checkbox(label, value=found[kind], disabled=not found[kind], key=f"export_{kind}"):
                    parts.add(kind)
        st.download_button(
            "Download settings",
            data=json.dumps(build_settings_bundle(parts), indent=2),
            file_name=f"rag-workspace-settings-{datetime.date.today():%Y-%m-%d}.json",
            mime="application/json",
            icon=":material/download:",
            key="settings_download",
        )

        st.markdown("**Import**")
        if done := st.session_state.get("settings_imported"):
            st.success("Restored " + ", ".join(done["restored"]) + f". Your previous settings are in `{done['backup']}`.")
            st.caption(
                "Next: make sure the notes are back in their folder, then index them. Reload the RAG workspace tab "
                "so it picks the settings up. If you use the connector's or exporter's schedule, turn it on again "
                "in that app (the import doesn't install schedules)."
            )
            with st.container(horizontal=True):
                if st.button("Index all corpora now", type="primary", icon=":material/bolt:", key="settings_index_now"):
                    index_all_corpora_here()
                st.button("Later", key="settings_index_later", on_click=st.session_state.pop,
                          args=("settings_imported", None),
                          help="Index later with **⚡ Rescan all corpora** in the RAG workspace's left panel.")
            return
        upload = st.file_uploader("Settings file", type=["json"], key="settings_import_file",
                                  help="A file from **Download settings**.")
        if upload is None:
            return
        bundle, problem = parse_settings_bundle(upload.getvalue())
        if bundle is None:
            st.error(problem)
            return
        st.caption(f"Exported {bundle.get('exported_at', '?')}. Restore:")
        chosen = {kind for kind, line in describe_settings_bundle(bundle).items()
                  if st.checkbox(line, value=True, key=f"import_{kind}")}
        st.caption("Each part replaces what's there now, at the path in the file; everything is backed up first.")
        if st.button("Import settings", type="primary", icon=":material/upload:", disabled=not chosen,
                     key="settings_import_apply"):
            restored, backup = apply_settings_bundle(bundle, chosen)
            st.session_state["settings_imported"] = {"restored": restored, "backup": str(backup)}
            st.rerun()


def render_configure_page() -> None:
    """Configure page (?page=configure): links, the outbox, the title marker, and export / import of settings."""
    library = load_prompt_library()
    render_outbox_status()
    render_page_title("Configure")
    st.caption(
        "Where Agents send their drafts and what processes them. The prompts themselves are in the "
        f"[Prompt library]({PROMPT_LIBRARY_PAGE_URL})."
    )

    links = library.get("links", [])
    with st.expander(f"Task executor settings · {len(links)} link{'' if len(links) == 1 else 's'}", expanded=True,
                     key="prompt_page_links_open", icon=":material/link:"):
        # The task executor comes first and is defined here; it can't be removed, only cleared.
        executor = executor_link(library) or {}
        st.markdown("**⚙️ Task executor**")
        st.caption(
            "The service that processes the outbox, e.g. the **Salesforce Connector** app "
            "(`http://localhost:8503`), where you run it now or set its schedule. It's shown at the top of the "
            "**Actions** page with 🟢 running / 🔴 not running, and you can attach it to prompts like any link."
        )
        with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
            st.text_input("Title", value=executor.get("title", ""), key="executor_title", on_change=save_executor_field,
                          args=("title",), max_chars=24, width=170, placeholder="Connector",
                          help="A word or two.")
            st.text_input("Address", value=executor.get("url", ""), key="executor_url", on_change=save_executor_field,
                          args=("url",), placeholder="http://localhost:8503")
            st.markdown(link_status_dot(executor.get("url", "")).strip() or " ", width="content",
                        help="🟢 running · 🔴 not running (local links only)")

        st.markdown("**🔗 Other links**")
        st.caption(
            "Links you attach to prompts with **Links** on each prompt card. While that prompt is injected they "
            "show under the question in the Ask box and open in a new tab. **Keep titles to a word or two** "
            "(e.g. *SF Tasks*): they sit inline. Put `{best_guess}` in an address for a **best guess search**: "
            "it's replaced by the app's best guess at what you're looking for (the Agent's *Related to* names, "
            "else the first **Must include** filter term, else names in your question). Links to this computer "
            "(localhost) show 🟢 when the service is running and 🔴 when it isn't."
        )
        for link in [link for link in links if link.get("role") != EXECUTOR_ROLE]:
            with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
                st.text_input(
                    "Title",
                    value=link.get("title", ""),
                    key=f"link_title_{link['id']}",
                    on_change=save_link_field,
                    args=(link["id"], "title"),
                    max_chars=24,
                    width=170,
                    placeholder="SF Tasks",
                    help="A word or two. It's shown inline under the question.",
                )
                st.text_input(
                    "Address",
                    value=link.get("url", ""),
                    key=f"link_url_{link['id']}",
                    on_change=save_link_field,
                    args=(link["id"], "url"),
                    placeholder="https://…  (may include {best_guess})",
                )
                st.markdown(link_status_dot(link.get("url", "")).strip() or " ", width="content",
                            help="🟢 running · 🔴 not running (local links only)")
                st.button(":material/close:", key=f"link_delete_{link['id']}", type="tertiary",
                          on_click=delete_link, args=(link["id"],), help="Remove this link, from every prompt too")
        st.button("Add link", icon=":material/add:", key="link_add", on_click=add_link)

    outbox = get_outbox_dir()
    waiting = len(pending_outbox_files(outbox))
    with st.expander(
        f"Outbox · {waiting} waiting",
        expanded=True,
        key="prompt_page_outbox_open",
        icon=":material/outbox:",
    ):
        st.caption(
            "Drafts you send from an **Agent** prompt are written here as JSON files. A separate connector "
            "(e.g. Salesforce) picks them up and moves each to processed/ or failed/ with a receipt."
        )
        st.text_input(
            "Outbox folder",
            value=str(outbox),
            key="outbox_path_input",
            on_change=persist_outbox_path,
            help="Must match the folder your connector reads. Leave empty for the default.",
        )
        counts = {folder: len(list((outbox / folder).glob("*.receipt.json"))) for folder in HANDOFF_RESULT_DIRS}
        st.caption(f"Waiting: **{waiting}** · Processed: **{counts['processed']}** · Failed: **{counts['failed']}**")

    marker = get_title_marker(library)
    with st.expander(
        "Task title marker · " + (f"“{marker['text']}”" if marker["text"] else "off"),
        expanded=True,
        key="prompt_page_title_marker_open",
        icon=":material/label:",
    ):
        st.caption(
            "Added to the title of everything an **Agent** stages (task subjects, note titles, email subjects), "
            "so you can find those records easily, e.g. in Salesforce. It's added when the Agent stages, so you "
            "see it before **That's right**; a title that already contains it is left alone. Empty means off."
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.text_input(
                "Marker",
                value=marker["text"],
                key="title_marker_text",
                on_change=save_title_marker,
                placeholder="e.g. RAG_TEST: or [AI]",
            )
            st.segmented_control(
                "Where",
                options=list(TITLE_MARKER_POSITIONS),
                default=marker["position"],
                format_func=TITLE_MARKER_POSITIONS.get,
                key="title_marker_position",
                on_change=save_title_marker,
            )
        if marker["text"]:
            st.caption(f"Example: **{marked_title('Follow up with Acme on the pilot', marker)}**")

    render_settings_transfer_section()
    install_save_flash()


def render_prompt_library_page() -> None:
    """Prompt library page (opened in its own tab): About me, corpus prompt, task prompts."""
    library = load_prompt_library()
    add_agent_border_style("[class*='st-key-agent_prompt_card_'] [data-testid='stExpander'] details")
    # Agent cards also get a blue hue, folded or open, so they read as agentic at a glance.
    st.html(f"<style>[class*='st-key-agent_prompt_card_'] [data-testid='stExpander'] details "
            f"{{ background-color: rgba({AGENT_TINT}, 0.08) !important; }}"
            f"[class*='st-key-agent_prompt_card_'] [data-testid='stExpander'] details > summary:hover "
            f"{{ background-color: rgba({AGENT_TINT}, 0.14) !important; }}</style>")
    render_outbox_status()
    render_page_title("Prompt library")
    st.caption(
        "Shape what the LLM is told when it answers from your notes. Each request is built in this order: "
        "today's date → About me → corpus prompt → global prompts → reference documents → your notes and question "
        "→ the task you pick. Links, the outbox and the title marker are on the "
        f"[Configure]({CONFIGURE_PAGE_URL}) page."
    )

    profile = library["profile"]
    profile_on = bool(profile.get("enabled", True) and str(profile.get("text", "")).strip())
    with st.expander(
        "About me · " + ("on" if profile_on else "off"),
        expanded=not str(profile.get("text", "")).strip(),
        key="prompt_page_about_me_open",
        icon=":material/person:",
    ):
        st.caption(
            "Sent with every LLM request in every corpus: who you are, where you're based, how you'd like "
            "answers. Today's date is always added for you."
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.checkbox(
                "Include in LLM requests",
                value=bool(library["profile"].get("enabled", True)),
                key="profile_enabled",
                on_change=save_profile_from_widgets,
                help="You can also leave it out of a single run from the Ask panel.",
            )
            st.text_input(
                "Short name",
                value=profile_short_name(library),
                key="profile_short_name",
                on_change=save_profile_from_widgets,
                max_chars=8,
                width=140,
                placeholder="e.g. JDOE",
                help=(
                    "Your initials or short name. It signs the updates Agents make, e.g. a Next Step reads "
                    "*2026-10-25-JDOE: waiting on the client…*. Not sent to the LLM."
                ),
            )
        st.text_area(
            "About me",
            value=str(library["profile"].get("text", "")),
            key="profile_text",
            on_change=save_profile_text,
            height=120,
            placeholder=(
                "e.g. I'm Sam Lee, a solutions architect at Example Ltd, based in London (UK time). "
                "I use a screen reader, so avoid tables and keep formatting simple."
            ),
            label_visibility="collapsed",
        )
        render_saved_note("profile_text")

    with st.expander(
        "Corpus prompt",
        expanded=False,
        key="prompt_page_corpus_prompt_open",
        icon=":material/folder:",
    ):
        st.caption("Describes who wrote the notes in one corpus. Only sent for that corpus.")
        libraries = get_libraries()
        if not libraries:
            st.info("Set your notes folder in the main app first.")
        else:
            last_library = load_last_selected_library()
            corpus = st.selectbox(
                "Corpus",
                libraries,
                index=libraries.index(last_library) if last_library in libraries else 0,
                key="prompt_page_corpus",
            )
            safe_corpus = resolve_library_registry_name(corpus)
            render_corpus_prompt_editor(safe_corpus, get_corpus_config(safe_corpus))

    quick_terms = (library.get("quick_search_terms") or [])[:QUICK_SEARCH_TERM_SLOTS]
    quick_terms += [""] * (QUICK_SEARCH_TERM_SLOTS - len(quick_terms))
    with st.expander(
        f"Quick search filters · {sum(1 for t in quick_terms if t.strip())} of {QUICK_SEARCH_TERM_SLOTS}",
        expanded=st.query_params.get("section") == "quick_search",
        key="prompt_page_quick_search_open",
        icon=":material/bolt:",
    ):
        st.caption(
            "Terms you filter by often, e.g. a customer or a folder such as `Clients-Prospects`. They show as "
            "buttons under the quick dates on each filter field (**Must include**, **Any of**, **Exclude**); a "
            "click adds the term to that field. Terms also match the folders notes are in. Saved as you type."
        )
        with st.container(horizontal=True, gap="small"):
            for i, term in enumerate(quick_terms):
                st.text_input(
                    f"Term {i + 1}",
                    value=term,
                    key=f"quick_search_term_{i}",
                    on_change=save_quick_search_terms,
                    max_chars=60,
                )

    render_reference_documents_section(library)

    with prompt_section("task", "Task prompts", ":material/checklist:"):
        st.caption(
            "Pick a task in the Ask panel to tell the LLM what to write from the notes. Tick **Global** to "
            "send a prompt with every request instead. Tick **Agent** to make it a workflow that ends by sending "
            "a note, tasks or an email to the outbox; Agent prompts have a blue border. **Model** picks the model "
            "used with the prompt. Tags help you find prompts; tick **Starred** for the ones you use most."
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            query = st.text_input(
                "Search prompts",
                key="prompt_page_query",
                placeholder="Full-text search of names, prompt text and tags",
                icon=":material/search:",
            )
            st.button(
                "Clear search",
                icon=":material/close:",
                on_click=set_prompt_page_query,
                args=("",),
                disabled=not query,
                key="prompt_page_clear_search",
            )
        keywords = common_prompt_tags(library)
        if keywords:
            with st.container(horizontal=True, vertical_alignment="center", gap="xsmall"):
                st.caption("Common keywords:")
                for keyword in keywords:
                    st.button(
                        keyword,
                        key=f"prompt_page_keyword_{keyword}",
                        type="primary" if query.strip().lower() == keyword.lower() else "secondary",
                        on_click=set_prompt_page_query,
                        args=(keyword,),
                    )
        shown = [p for p in library["prompts"] if prompt_matches_query(p, query)]
        # "Edit prompt workflow" in the Discussion panel opens this page with ?prompt=<id>.
        focused = find_prompt(library, st.query_params.get("prompt"))
        if focused:
            shown = [focused]
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(f"Showing the workflow you opened: **{focused['name']}**.")
                st.button("Show all prompts", key="prompt_page_show_all", on_click=show_all_prompts)
            st.session_state[f"prompt_open_{focused['id']}"] = True
        elif query.strip():
            st.caption(f"{len(shown)} of {len(library['prompts'])} prompts match.")
        model_options = prompt_model_options()
        # The Agent tick decides the section: ticking or unticking it moves the prompt at once.
        tasks = [p for p in shown if not is_agent_prompt(p)]
        render_prompt_section_cards("task", tasks, library, model_options,
                                    "No task prompts match." if library["prompts"] else "No prompts yet — add one below.")

    # The Agent section's own expander (not the cards inside it) gets the blue outline and heading.
    st.html(f"<style>.st-key-agent_prompts {{ --section: {AGENT_TINT}; }}"
            f".st-key-agent_prompts details:not(details details) {{ border: 1px solid rgb({AGENT_TINT}) !important; }}"
            f".st-key-agent_prompts details:not(details details) > summary {{ color: rgb({AGENT_TINT}); "
            f"font-weight: 600; }}{section_shading_css()}</style>")
    with prompt_section("agent", "Agent prompts", ":material/smart_toy:"):
        st.caption(
            "Prompts with **Agent** ticked: workflows that end by sending a note, tasks, an email or a Next Step "
            "to the outbox. Untick **Agent** to move one to Task prompts. The search above covers both sections."
        )
        agents = [p for p in shown if is_agent_prompt(p)]
        render_prompt_section_cards("agent", agents, library, model_options, "No Agent prompts match.")

    render_semantic_fixes_section(library)

    with st.container(border=True):
        st.subheader("Add a prompt")
        with st.form("add_prompt_form", clear_on_submit=True, border=False):
            st.text_input("Name", key="new_prompt_name", placeholder="e.g. Proposal outline")
            st.text_area(
                "Prompt",
                key="new_prompt_text",
                height=100,
                placeholder="What should the LLM do with the notes?",
            )
            with st.container(horizontal=True, vertical_alignment="center"):
                st.checkbox("Global", key="new_prompt_global", help="Send with every LLM request in every corpus.")
                st.checkbox("Starred", key="new_prompt_starred", help="Show it in the Ask panel's Starred row (Agents: its Agent Starred row).")
                st.checkbox(
                    "Agent",
                    key="new_prompt_agent",
                    help="Draft a note, tasks or an email to review and send to the outbox, instead of a text answer.",
                )
                st.selectbox(
                    "Output",
                    options=DRAFT_OUTPUT_TYPES,
                    index=DRAFT_OUTPUT_TYPES.index(DEFAULT_AGENT_OUTPUT),
                    format_func=OUTPUT_TYPES.get,
                    key="new_prompt_output",
                    width=150,
                    label_visibility="collapsed",
                    help="What an Agent prompt drafts. Ignored unless **Agent** is ticked.",
                )
                st.selectbox(
                    "Model",
                    options=model_options,
                    format_func=format_prompt_model,
                    key="new_prompt_model",
                    width=200,
                    label_visibility="collapsed",
                    help="Model to use when you pick this prompt. You can still change it in the Ask panel.",
                )
                st.multiselect(
                    "Tags",
                    options=[],
                    key="new_prompt_tags",
                    accept_new_options=True,
                    placeholder="Add a tag, press Enter",
                    label_visibility="collapsed",
                )
            references = reference_documents(library)
            if references:
                ref_titles = {ref["id"]: "📄 " + reference_label(ref) for ref in references}
                st.multiselect(
                    "Reference documents",
                    options=list(ref_titles),
                    format_func=ref_titles.get,
                    key="new_prompt_references",
                    placeholder="Reference documents to send with this prompt",
                    help="Sent to the LLM as background whenever this prompt is used, Agent or not. You can change "
                    "them later on the prompt's card.",
                )
            else:
                st.caption("No reference documents yet: make them in **Reference documents** above, then attach "
                           "them here or on the prompt's card.")
            st.checkbox("☁️ Send Salesforce records", key="new_prompt_salesforce", help=SALESFORCE_PROMPT_HELP)
            st.form_submit_button("Add prompt", type="primary", icon=":material/add:", on_click=add_prompt_from_form)
        if error := st.session_state.pop("add_prompt_error", None):
            st.warning(error)
        if added := st.session_state.pop("add_prompt_added", None):
            st.success(f"Added **{added}**. Agent prompts get their steps and Action Command on their card above.")

    install_filter_commit_on_blur()
    install_tooltip_hide_on_click()
    install_save_flash()


ACTIVE_TASK_PROMPTS_KEY = "active_task_prompt_ids"


def active_task_prompts(library: dict) -> list[dict]:
    """The injected task prompts: one plain prompt, or one or more Agents (e.g. SF_Next_Step and SF_Tasks)."""
    prompts = (find_prompt(library, prompt_id) for prompt_id in st.session_state.get(ACTIVE_TASK_PROMPTS_KEY, []))
    return [p for p in prompts if p and not p.get("global")]


def set_active_task_prompt(prompt_id: str) -> None:
    """Inject a task prompt. Agents add up, so one action can update Next Steps and create tasks; a plain
    prompt replaces whatever was injected, and an Agent replaces a plain prompt."""
    library = load_prompt_library()
    current = active_task_prompts(library)
    if is_agent_prompt(find_prompt(library, prompt_id)) and all(is_agent_prompt(p) for p in current):
        ids = [p["id"] for p in current if p["id"] != prompt_id] + [prompt_id]
    else:
        ids = [prompt_id]
    st.session_state[ACTIVE_TASK_PROMPTS_KEY] = ids


def remove_active_task_prompt(prompt_id: str) -> None:
    ids = st.session_state.get(ACTIVE_TASK_PROMPTS_KEY, [])
    st.session_state[ACTIVE_TASK_PROMPTS_KEY] = [i for i in ids if i != prompt_id]


def sync_injected_prompt_token() -> None:
    st.session_state[ACTIVE_TASK_PROMPTS_KEY] = list(st.session_state.get("question_task_token") or [])


def render_injected_prompt_token(library: dict) -> None:
    """Red tokens under the question box showing the injected task prompts; each × removes that one."""
    tasks = active_task_prompts(library)
    if not tasks:
        return
    by_id = {task["id"]: task for task in tasks}
    st.session_state["question_task_token"] = list(by_id)
    if any(is_agent_prompt(task) for task in tasks):
        add_agent_border_style(".st-key-question_task_token [data-tag]")
    st.multiselect(
        "Injected prompts",
        options=list(by_id),
        format_func=lambda prompt_id: f"Prompt: {agent_prompt_label(by_id[prompt_id])}",
        key="question_task_token",
        on_change=sync_injected_prompt_token,
        label_visibility="collapsed",
        help="Injected into the LLM request as the task" + (
            "s, one part each" if len(tasks) > 1 else ""
        ) + ":\n\n" + "\n\n".join(f"**{task['name']}:** {task['text']}" for task in tasks),
    )


def pick_task_prompt_from_finder() -> None:
    prompt_id = st.session_state.get("task_prompt_finder")
    if prompt_id:
        set_active_task_prompt(prompt_id)
    st.session_state["task_prompt_finder"] = None


def resolve_run_prompts(library: dict, corpus_config: dict) -> dict:
    """What this run sends to the LLM, honouring the per-run 'leave out' ticks in the Ask panel."""
    profile = library["profile"]
    profile_text = str(profile.get("text", "")).strip() if profile.get("enabled") else ""
    corpus_prompt = get_active_corpus_prompt(corpus_config)
    global_prompts = [p for p in library["prompts"] if p.get("global") and p.get("text", "").strip()]
    return {
        "profile": profile_text if st.session_state.get("run_include_profile", True) else "",
        "corpus": corpus_prompt if st.session_state.get("run_include_corpus", True) else "",
        "globals": [p for p in global_prompts if st.session_state.get(f"run_include_global_{p['id']}", True)],
        "tasks": active_task_prompts(library),
    }


def first_prompt_model(prompts: list[dict]) -> str | None:
    """The model the first of these prompts asks for, if any (a prompt can pick its own model)."""
    return next((p["model"] for p in prompts if p.get("model")), None)


def describe_run_prompts(run_prompts: dict, corpus_prompt: str) -> list[str]:
    parts = []
    if run_prompts["profile"]:
        parts.append("About me")
    if corpus_prompt:
        parts.append("corpus prompt")
    parts.extend(f"Global: {p['name']}" for p in run_prompts["globals"])
    parts.extend(f"Task: {agent_prompt_label(task)}" for task in run_prompts["tasks"])
    return parts


# The Recent and Starred prompt buttons, about a quarter smaller than a normal button; the text keeps its size.
PROMPT_CHIP_STYLE = """<style>
[class*="st-key-use_prompt_"] button, [class*="st-key-use_agent_prompt_"] button,
[class*="st-key-use_reference_"] button, .st-key-use_salesforce_context button {
  min-height: 1.875rem !important; padding: 0.05rem 0.55rem !important;
}
/* Section shading paints every button alike, so a chip that's switched on gets a firm outline and bold text. */
.st-key-ask_reference_row button[data-testid="stBaseButton-primary"],
[class*="_add_context"] button[data-testid="stBaseButton-primary"] {
  box-shadow: inset 0 0 0 2px currentColor !important; font-weight: 600 !important;
}
</style>"""


PROMPT_ROW_CHARS = 52  # about what fits on the Recent or Starred row after its label; chips that don't are left out


def fit_prompt_chips(prompts: list[dict], row_chars: int = PROMPT_ROW_CHARS) -> list[tuple[dict, str]]:
    """(prompt, chip label) for as many prompts as fit on one line, in order, skipping any that would overflow."""
    shown, width = [], 0
    for prompt in prompts:
        label = truncate_text(agent_prompt_label(prompt), 34 if is_agent_prompt(prompt) else 28)
        cost = len(label) + 4  # a chip's padding and the gap
        if width + cost <= row_chars:
            shown.append((prompt, label))
            width += cost
    return shown


def set_corpus_prompt_enabled(safe_lib: str) -> None:
    """The Ask box's Corpus prompt toggle: the same per-corpus setting as Apply corpus prompt."""
    save_corpus_config(safe_lib, {"corpus_prompt_enabled": bool(st.session_state.get(f"ask_corpus_prompt_{safe_lib}"))})


def render_ask_prompt_panel(library: dict, corpus_config: dict, safe_lib: str = "") -> None:
    """Compact 'Prompt library' box in the Ask panel: always-included prompts, task, recent and starred."""
    profile = library["profile"]
    profile_text = str(profile.get("text", "")).strip() if profile.get("enabled") else ""
    corpus_prompt = get_active_corpus_prompt(corpus_config)
    global_prompts = [p for p in library["prompts"] if p.get("global") and p.get("text", "").strip()]
    task_prompts = [p for p in library["prompts"] if not p.get("global")]
    active = resolve_run_prompts(library, corpus_config)["tasks"]
    active_ids = {task["id"] for task in active}
    add_agent_border_style("[class*='st-key-use_agent_prompt_'] button", "[class*='st-key-active_agent_task']")
    # Agent chips (Recent, Starred, Agent Starred) are shaded blue, deeper when chosen, whatever section they're in.
    chip = "[class*='st-key-use_agent_prompt_'][class*='st-key-use_agent_prompt_'] button"
    st.html(f"<style>[class*='st-key-use_agent_prompt_'] {{ --section: {AGENT_TINT}; }}"
            f"{chip} {{ background-color: rgba({AGENT_TINT}, 0.22) !important; color: inherit !important; }}"
            f"{chip}:hover {{ background-color: rgba({AGENT_TINT}, 0.36) !important; }}"
            f"{chip}[data-testid='stBaseButton-primary'] {{ background-color: rgba({AGENT_TINT}, 0.55) !important; "
            "font-weight: 600 !important; }</style>")
    st.html("<style>[class*='st-key-active_agent_task'] { padding: 0.1rem 0.25rem 0.1rem 0.75rem !important; }</style>")
    st.html(PROMPT_CHIP_STYLE)

    with st.container(border=True, gap="xsmall"):
        with st.container(horizontal=True, vertical_alignment="center", gap="xsmall", key="prompt_panel_title"):
            st.markdown("**Prompt library**", width="content")
            # Icon only, so About me and Corpus prompt fit on this line.
            st.link_button(
                ":material/open_in_new:",
                PROMPT_LIBRARY_PAGE_URL,
                type="tertiary",
                help="Manage: opens the prompt library in a new tab.",
            )
            if profile_text:
                st.toggle("About me", value=True, key="run_include_profile", help=profile_text)
            # Kept in step with the saved setting, which the Prompt library page can change too.
            corpus_text = str(corpus_config.get("corpus_prompt", "")).strip()
            corpus_key = f"ask_corpus_prompt_{safe_lib}"
            st.session_state[corpus_key] = bool(corpus_config.get("corpus_prompt_enabled")) and bool(corpus_text)
            st.toggle(
                "Corpus prompt",
                key=corpus_key,
                on_change=set_corpus_prompt_enabled,
                args=(safe_lib,),
                disabled=not corpus_text,
                help=(corpus_text + "\n\nSent ahead of the notes for every answer and follow-up in this corpus. "
                      "Saved per corpus." if corpus_text
                      else "This corpus has no corpus prompt yet: write one on the Prompt library page (the ↗ beside **Prompt library**)."),
            )

        if global_prompts:
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.caption("Always:")
                for prompt in global_prompts:
                    st.checkbox(
                        prompt["name"], value=True, key=f"run_include_global_{prompt['id']}", help=prompt["text"]
                    )

        recent, starred = select_prompt_stacks(library)
        recent_chips = fit_prompt_chips(recent)
        in_recent = {prompt["id"] for prompt, _label in recent_chips}
        starred = [prompt for prompt in starred if prompt["id"] not in in_recent]
        # The longer "Agent Starred:" label leaves less room for chips.
        agent_row_chars = PROMPT_ROW_CHARS - (len("Agent Starred") - len("Starred"))
        rows = (
            ("Recent", recent_chips),
            ("Starred", fit_prompt_chips([p for p in starred if not is_agent_prompt(p)])),
            ("Agent Starred", fit_prompt_chips([p for p in starred if is_agent_prompt(p)], agent_row_chars)),
        )
        for title, chips in rows:
            if not chips:
                continue
            with st.container(horizontal=True, vertical_alignment="center", gap="xsmall"):
                st.caption(f"{title}:", width="content")
                # Only what fits on this line; the rest are in Find a prompt.
                for prompt, label in chips:
                    is_active = prompt["id"] in active_ids
                    agent = is_agent_prompt(prompt)
                    st.button(
                        label,
                        # Agent keys pick up the blue border style above.
                        key=f"use_{'agent_' if agent else ''}prompt_{title.replace(' ', '')}_{prompt['id']}",
                        help=(f"**Agent** — ends by sending {OUTPUT_TYPES[get_prompt_output_type(prompt)].lower()} "
                              f"to the outbox: *{agent_command(prompt)}*\n\n" if agent else "") + prompt["text"],
                        type="primary" if is_active else "secondary",
                        on_click=remove_active_task_prompt if is_active else set_active_task_prompt,
                        args=(prompt["id"],),
                    )

        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            for task in active:
                agent = is_agent_prompt(task)
                with st.container(
                    horizontal=True,
                    vertical_alignment="center",
                    gap="xsmall",
                    width="content",
                    border=agent,
                    key=f"active_{'agent_' if agent else ''}task_{task['id']}",
                ):
                    st.markdown(f"**Task:** {agent_prompt_label(task)}", help=task["text"], width="content")
                    st.button(
                        ":material/close:",
                        type="tertiary",
                        key=f"clear_task_prompt_{task['id']}",
                        on_click=remove_active_task_prompt,
                        args=(task["id"],),
                        help="Remove this task",
                    )
            if task_prompts:
                # A dropdown can't style single options, so Agents show what they draft instead.
                labels = {p["id"]: agent_prompt_label(p) for p in task_prompts}
                st.selectbox(
                    "Find a prompt",
                    options=list(labels),
                    index=None,
                    format_func=labels.get,
                    key="task_prompt_finder",
                    on_change=pick_task_prompt_from_finder,
                    placeholder="Find a prompt…" if active else "No task — find a prompt…",
                    label_visibility="collapsed",
                )
            else:
                st.caption("No task prompts yet — add some with **Manage**.")
        render_ask_reference_row(library, active)


def render_prompt_library_sidebar(corpus_config: dict) -> None:
    """Sidebar button that opens the Prompt library page; hover it for a summary of what is sent to the LLM."""
    library = load_prompt_library()
    profile = library["profile"]
    global_count = sum(1 for p in library["prompts"] if p.get("global"))
    task_count = len(library["prompts"]) - global_count
    agent_count = sum(1 for p in library["prompts"] if is_agent_prompt(p))
    st.sidebar.link_button(
        "Prompt library",
        PROMPT_LIBRARY_PAGE_URL,
        icon=":material/library_books:",
        width="stretch",
        help=(
            f"About me: **{'on' if profile.get('enabled') and profile.get('text', '').strip() else 'off'}** · "
            f"Corpus prompt: **{'applied' if get_active_corpus_prompt(corpus_config) else 'off'}**  \n"
            f"Global prompts: **{global_count}** · Task prompts: **{task_count}** (Agents: **{agent_count}**)  \n"
            "Opens in a new tab."
        ),
    )
    st.sidebar.link_button(
        "Configure",
        CONFIGURE_PAGE_URL,
        icon=":material/settings:",
        width="stretch",
        help="Task executor settings (the connector and other links), the outbox and the task title marker. "
        "Opens in a new tab.",
    )


def is_hide_corpora_enabled() -> bool:
    return bool(
        st.session_state.get(
            "hide_corpora_enabled_global",
            load_app_settings().get("hide_corpora_enabled", False),
        )
    )


def persist_hide_corpora_enabled() -> None:
    save_app_settings(
        {"hide_corpora_enabled": bool(st.session_state.get("hide_corpora_enabled_global", False))}
    )


def is_corpus_hidden(library_name: str) -> bool:
    return bool(get_corpus_config(sanitize_library_name(library_name)).get("hidden", False))


def filter_libraries_for_display(libraries: list[str]) -> list[str]:
    if not load_app_settings().get("hide_corpora_enabled", False):
        return libraries
    return [lib for lib in libraries if not is_corpus_hidden(lib)]


def persist_corpus_hidden_toggle(library_name: str) -> None:
    safe = sanitize_library_name(library_name)
    save_corpus_config(safe, {"hidden": bool(st.session_state.get(f"corpus_hidden_{safe}", False))})


def render_corpus_visibility_controls(all_libraries: list[str]) -> None:
    """Corpus hide controls tucked away so Advanced stays demo-friendly."""

    def _render_controls() -> None:
        st.checkbox(
            "Enable corpus hiding in library list",
            value=bool(load_app_settings().get("hide_corpora_enabled", False)),
            key="hide_corpora_enabled_global",
            on_change=persist_hide_corpora_enabled,
            help=(
                "When enabled, corpora marked hidden below are removed from the library dropdown and "
                "skipped by all-corpora scans and rebuilds."
            ),
        )
        if is_hide_corpora_enabled():
            st.caption(
                "Hidden corpora are removed from the selector and skipped by every all-corpora scan "
                "and rebuild (including auto-rescan). Their index is kept as it was; select one to "
                "rescan it on its own."
            )
            for lib_name in all_libraries:
                safe_name = sanitize_library_name(lib_name)
                st.checkbox(
                    f"Hide `{lib_name}`",
                    value=bool(get_corpus_config(safe_name).get("hidden", False)),
                    key=f"corpus_hidden_{safe_name}",
                    on_change=persist_corpus_hidden_toggle,
                    args=(lib_name,),
                )

    if hasattr(st, "popover"):
        with st.popover("Manage corpus visibility…"):
            _render_controls()
    else:
        show_controls = st.checkbox(
            "Show corpus visibility controls",
            value=False,
            key="show_corpus_visibility_controls",
        )
        if show_controls:
            _render_controls()


def load_all_job_statuses() -> dict:
    if not JOB_STATUS_PATH.exists():
        return {}
    try:
        return json.loads(JOB_STATUS_PATH.read_text())
    except Exception:
        return {}


def get_persisted_job_status(safe_lib: str) -> dict | None:
    status = load_all_job_statuses().get(safe_lib)
    return status if isinstance(status, dict) else None


def save_persisted_job_status(safe_lib: str, status: dict) -> dict:
    all_statuses = load_all_job_statuses()
    all_statuses[safe_lib] = status
    JOB_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOB_STATUS_PATH.write_text(json.dumps(all_statuses, indent=2))
    return status


def commit_job_status(status: dict, selected_lib: str, safe_lib: str) -> dict:
    """Persist scan results and keep them in session for the sidebar."""
    full_status = {**status, "library_display": selected_lib, "safe_lib": safe_lib}
    save_persisted_job_status(safe_lib, full_status)
    st.session_state.job_status = full_status
    return full_status


def resolve_job_status(selected_lib: str, safe_lib: str) -> dict | None:
    """Return the latest scan status for this corpus (session first, then disk)."""
    session_status = st.session_state.get("job_status")
    if session_status and session_status.get("library_display") == selected_lib:
        return session_status
    return get_persisted_job_status(safe_lib)


def render_sidebar_job_summary(selected_lib: str, safe_lib: str) -> None:
    """Show the last completed ingest/embed job with a persistent progress bar."""
    status = resolve_job_status(selected_lib, safe_lib)
    if not status:
        return

    progress_text = status.get("progress_text") or status.get("message") or "Scan complete"
    progress_pct = float(status.get("progress_pct", 1.0))
    embed_current = status.get("embedding_current")
    embed_total = status.get("embedding_total")
    operation = status.get("operation", "Scan")
    completed_at = status.get("completed_at", "unknown")

    st.markdown("**Corpus index**")
    st.progress(min(max(progress_pct, 0.0), 1.0), text=progress_text)
    st.success(progress_text)
    st.caption(f"{operation} · {completed_at}")

    detail_parts = [
        f"Docs added: **{status.get('docs_added', 0)}**",
        f"Blocks added: **{status.get('blocks_added', 0)}**",
        f"Total blocks: **{status.get('blocks_total', 0)}**",
        f"Vectors: **{status.get('vectors', 0)}**",
    ]
    st.caption(" · ".join(detail_parts))

    meta_parts = [f"Model: `{status.get('embedding_model', '?')}`"]
    if embed_total:
        meta_parts.append(f"Embedded: **{embed_current}/{embed_total}** blocks")
    meta_parts.append(f"Folders scanned: **{status.get('dirs_scanned', 0)}**")
    st.caption(" · ".join(meta_parts))


FOLDER_PATH_SEPARATOR = " > "


@st.cache_data(ttl=300, show_spinner=False)
def library_file_folders(folder: str) -> dict[str, list[str]]:
    """{file name: [its folder, relative to `folder`]} for every note under a library folder (cached 5 min)."""
    root = Path(folder)
    index: dict[str, list[str]] = {}
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in INGEST_EXTENSIONS and not path.name.startswith("."):
                index.setdefault(path.name, []).append(str(path.parent.relative_to(root)))
    return index


def note_folder_path(file_source: str, library_folder: Path, corpus_root: Path | None) -> str:
    """Where a note lives, from its file's location, e.g. "Clients > Prospects > Acme".

    The index's own copy of a file is flat, so the note is found by name in its library folder. Empty when
    it can't be found, or when two notes in the library share the name (rather than guess).
    """
    if not file_source or not library_folder:
        return ""
    matches = library_file_folders(str(library_folder)).get(Path(str(file_source)).name, [])
    if len(matches) != 1:
        return ""
    folder = (Path(library_folder) / matches[0]).resolve()
    # Relative to the corpus root, so multi-library paths start with the library ("Clients > …").
    base = Path(corpus_root).resolve() if corpus_root else Path(library_folder).resolve().parent
    if not folder.is_relative_to(base):
        base = Path(library_folder).resolve().parent
    return FOLDER_PATH_SEPARATOR.join(part for part in folder.relative_to(base).parts if part not in ("", "."))


@st.cache_data(ttl=300, show_spinner=False)
def library_source_paths(safe_lib: str, library_folder: str, corpus_root: str) -> dict[str, str]:
    """{file_source: folder path} for every note in a library's index whose folder is known (cached 5 min)."""
    table = safe_lib.replace('"', '""')
    try:
        with sqlite3.connect(SQLiteConfig.get_uri_string()) as conn:
            sources = [row[0] for row in conn.execute(f'SELECT DISTINCT file_source FROM "{table}"')]
    except sqlite3.Error:
        return {}
    root = Path(corpus_root) if corpus_root else None
    paths = {str(source): note_folder_path(str(source), Path(library_folder), root) for source in sources}
    return {source: path for source, path in paths.items() if path}


def with_path_matches(search_filter: dict, safe_lib: str, library_folder: Path | None, corpus_root: Path | None) -> dict:
    """Let filter terms match the folders notes are in, e.g. `Clients-Prospects`.

    A term matches a path written either way, "Clients > Prospects" or "Clients/Prospects".
    """
    terms = {term.lower() for field in SEARCH_FILTER_FIELDS for term in search_filter.get(field, [])}
    if not terms or not library_folder:
        return search_filter
    paths = library_source_paths(safe_lib, str(library_folder), str(corpus_root or ""))
    haystacks = {
        source: f"{path}\n{path.replace(FOLDER_PATH_SEPARATOR, '/')}".lower() for source, path in paths.items()
    }
    path_sources = {}
    for term in terms:
        if hits := [source for source, haystack in haystacks.items() if term in haystack]:
            path_sources[term] = hits
    return {**search_filter, "path_sources": path_sources}


def add_folder_paths(results: list, library_folder: Path, corpus_root: Path | None) -> None:
    """Give each search result its note's folder path (`_folder_path`), for the LLM and the Sources list."""
    for result in results:
        result["_folder_path"] = note_folder_path(
            str(result.get("file_source", "")), result.get("_library_folder", library_folder), corpus_root
        )


def resolve_source_path(
    file_source: str, library_folder: Path, extra_dirs: tuple[Path, ...] = ()
) -> Path | None:
    """Map a llmware file_source value to a local file path."""
    if not file_source:
        return None

    source = Path(str(file_source)).expanduser()
    if source.is_file():
        return source.resolve()

    if source.is_absolute() and source.exists():
        return source.resolve()

    by_name = library_folder / source.name
    if by_name.is_file():
        return by_name.resolve()

    relative = library_folder / source
    if relative.is_file():
        return relative.resolve()

    filename = source.name or str(file_source).split("/")[-1]
    matches = [path for path in library_folder.rglob(filename) if path.is_file()]
    if len(matches) == 1:
        return matches[0].resolve()

    # Fall back to the copy llmware keeps in the library uploads folder.
    for directory in extra_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()

    return None


TEXT_PREVIEW_EXTENSIONS = {".md", ".txt", ".csv", ".json"}
DOCUMENT_PREVIEW_MAX_CHARS = 200_000


@st.cache_data(ttl=300, show_spinner=False)
def load_document_text_from_index(safe_lib: str, file_source: str) -> str:
    """Rebuild a document's text from its indexed blocks (works for PDF/DOCX/PPTX too)."""
    db_path = SQLiteConfig.get_uri_string()
    table = safe_lib.replace('"', '""')
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                f'SELECT text_block FROM "{table}" WHERE file_source = ? '
                "ORDER BY doc_ID, block_ID",
                (file_source,),
            ).fetchall()
    except sqlite3.Error:
        return ""
    return "\n\n".join(row[0] for row in rows if row[0])


def get_library_uploads_dirs(lib: Library | None) -> tuple[Path, ...]:
    """Folders where llmware keeps copies of ingested files."""
    if lib is None:
        return ()
    try:
        return (Path(lib.file_copy_path),)
    except Exception:
        return ()


def render_source_document(
    file_source: str,
    hits: list[dict],
    library_folder: Path,
    safe_lib: str,
    extra_dirs: tuple[Path, ...] = (),
) -> None:
    """Show matched passages and the full embedded document inside the current container."""
    path = resolve_source_path(file_source, library_folder, extra_dirs)
    st.caption(str(path) if path else f"Source: `{file_source}`")

    passages_tab, document_tab = st.tabs(["Matched passages", "Full document"])

    with passages_tab:
        for hit in hits:
            distance = hit.get("distance")
            if distance is not None:
                st.caption(f"Result #{hit['_rank']} · distance {distance:.3f}")
            else:
                st.caption(f"Result #{hit['_rank']}")
            st.markdown(f"> {str(hit.get('text', '')).strip()}".replace("\n", "\n> "))

    with document_tab:
        render_full_document(file_source, path, safe_lib)


def render_full_document(file_source: str, path: Path | None, safe_lib: str, where: str = "sources") -> None:
    """Full note text (from disk, or rebuilt from the index) plus a download button.

    `where` keeps the button's key unique when a note is shown twice (gathered notes and an answer's sources).
    """
    text = ""
    from_index = False
    if path and path.suffix.lower() in TEXT_PREVIEW_EXTENSIONS:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
    if not text:
        text = load_document_text_from_index(safe_lib, Path(file_source).name)
        from_index = bool(text)

    if not text:
        st.info("Document contents are not available.")
    else:
        if from_index:
            st.caption("Text reconstructed from the indexed blocks.")
        truncated = len(text) > DOCUMENT_PREVIEW_MAX_CHARS
        preview = text[:DOCUMENT_PREVIEW_MAX_CHARS]
        with st.container(height=500):
            if path and path.suffix.lower() == ".md":
                st.markdown(preview)
            elif path and path.suffix.lower() == ".json":
                st.code(preview, language="json")
            else:
                st.text(preview)
        if truncated:
            st.caption(f"Preview truncated to {DOCUMENT_PREVIEW_MAX_CHARS:,} characters.")

    if path and path.is_file():
        try:
            st.download_button(
                f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                key=f"download::{where}::{safe_lib}::{file_source}",
            )
        except OSError:
            pass


def describe_prompt_parts(search_state: dict) -> str:
    """Caption suffix naming the prompts sent with this search, e.g. " with About me, Task: Action items."."""
    parts = search_state.get("prompt_parts") or []
    return f" with {', '.join(parts)}." if parts else "."


def discussion_note_key(result: dict) -> tuple[str, str]:
    return (result.get("_library", ""), str(result.get("file_source", "")))


def discussion_results(search_state: dict) -> list:
    """The search's results minus notes the user unticked under Sources ('use in this discussion')."""
    left_out = {tuple(key) for key in search_state.get("left_out_notes", [])}
    return [r for r in search_state["results"] if discussion_note_key(r) not in left_out]


def toggle_discussion_note(search_state: dict, note_key: tuple[str, str], widget_key: str) -> None:
    left_out = [tuple(key) for key in search_state.get("left_out_notes", []) if tuple(key) != note_key]
    if not st.session_state.get(widget_key, True):
        left_out.append(note_key)
    search_state["left_out_notes"] = left_out


def build_discussion_messages(search_state: dict, max_tokens: int, temperature: float, corpus_prompt: str) -> list[dict]:
    """The conversation so far, opening with the notes still in use (rebuilt each time, so unticking applies)."""
    messages = [{
        "role": "user",
        "content": build_llm_prompt(
            search_state["question"],
            discussion_results(search_state),
            max_tokens,
            temperature,
            with_salesforce_context(search_state, corpus_prompt, discussion_results(search_state)),
            search_state.get("task_prompt", ""),
        ),
    }]
    if search_state.get("answer"):
        messages.append({"role": "assistant", "content": search_state["answer"]})
    messages.extend(
        {"role": turn["role"], "content": turn.get("prompt", turn["content"])}
        for turn in search_state["discussion"]
    )
    return messages


def follow_up_with_ollama(
    search_state: dict,
    question: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str = "",
    sent_text: str | None = None,
) -> dict:
    """Answer a follow-up using the notes in use plus the conversation so far.

    `sent_text` is sent instead of `question` when they differ (e.g. a step's full prompt); it is kept
    as the turn's `prompt` so later requests repeat what the model actually saw.
    """
    messages = build_discussion_messages(search_state, max_tokens, temperature, corpus_prompt)
    user_turn = {"role": "user", "content": question}
    if sent_text and sent_text != question:
        user_turn["prompt"] = sent_text
    messages.append({"role": "user", "content": user_turn.get("prompt", question)})

    reply, _usage = chat_with_ollama(
        messages,
        model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        num_ctx=FOLLOW_UP_NUM_CTX,
    )
    return {"user": user_turn, "assistant": {"role": "assistant", "content": reply}}


def add_discussion_turn(
    search_state: dict,
    question: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str,
    sent_text: str | None = None,
) -> bool:
    """Ask the follow-up and add both turns to the discussion; False (with an error shown) if it failed."""
    with st.spinner(f"Asking `{model_name}`…"):
        try:
            turns = follow_up_with_ollama(
                search_state, question, model_name, max_tokens, temperature, corpus_prompt, sent_text
            )
        except Exception as exc:
            st.error(f"Follow-up failed: {exc}")
            return False
    turns["assistant"]["model"] = model_name
    search_state["discussion"].extend([turns["user"], turns["assistant"]])
    return True


def persist_llm_tuning(safe_lib: str) -> None:
    """Save Max tokens / Temperature as soon as a slider moves, so they survive new sessions."""
    updates = {}
    if f"llm_max_tokens_{safe_lib}" in st.session_state:
        updates["llm_max_tokens"] = int(st.session_state[f"llm_max_tokens_{safe_lib}"])
    if f"llm_temperature_{safe_lib}" in st.session_state:
        updates["llm_temperature"] = float(st.session_state[f"llm_temperature_{safe_lib}"])
    if updates:
        save_corpus_config(safe_lib, updates)


def persist_follow_up_model(safe_lib: str) -> None:
    save_corpus_config(safe_lib, {"follow_up_model": st.session_state.get(f"follow_up_model_{safe_lib}")})


def set_big_screen(search_state: dict, choice: int | str | None) -> None:
    """Show a discussion reply in the main output area: a turn index, "last" (follows new replies) or None."""
    search_state["big_screen"] = choice


def big_screen_turn(search_state: dict) -> tuple[int, dict] | None:
    """The discussion reply chosen for the main output area, as (index, turn), or None for the answer."""
    choice = search_state.get("big_screen")
    discussion = search_state.get("discussion") or []
    replies = [i for i, turn in enumerate(discussion) if turn["role"] == "assistant"]
    if choice == "last" and replies:
        return replies[-1], discussion[replies[-1]]
    if isinstance(choice, int) and choice in replies:
        return choice, discussion[choice]
    return None


def render_big_screen_reply(search_state: dict, index: int, turn: dict) -> None:
    """A discussion reply in the main output area, with the question that led to it and the way back."""
    discussion = search_state["discussion"]
    replies = [i for i, t in enumerate(discussion) if t["role"] == "assistant"]
    number = replies.index(index) + 1
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Discussion reply", width="content")
        render_copy_buttons([("Copy all output", full_llm_output(search_state))])
    following = " · follows new replies" if search_state.get("big_screen") == "last" else ""
    model = f" · `{turn['model']}`" if turn.get("model") else ""
    st.caption(f"Reply {number} of {len(replies)} from the discussion{model}{following}")
    asked = discussion[index - 1]["content"] if index and discussion[index - 1]["role"] == "user" else ""
    if asked:
        with st.expander("Question asked"):
            st.markdown(asked)
    st.markdown(link_opportunity_names(turn["content"]))
    render_copy_buttons([("Copy reply", link_opportunity_names(turn["content"])),
                         ("Copy all output", full_llm_output(search_state))])
    st.button(
        "↩ Back to the answer",
        key="big_screen_back",
        on_click=set_big_screen,
        args=(search_state, None),
        help="Show the first answer here again; the reply stays in the discussion.",
    )


def render_copy_button(text: str, label: str = "Copy") -> None:
    """Small button that copies `text` to the clipboard (component iframes may write to it)."""
    render_copy_buttons([(label, text)])


def full_llm_output(search_state: dict) -> str:
    """Everything the LLM wrote for this search, as Markdown: the question, the answer and the discussion."""
    parts = []
    if search_state.get("question"):
        parts.append("## Question\n\n" + search_state["question"].strip())
    # Copied text carries the opportunity links the app adds on screen.
    parts.append("## Answer\n\n" + link_opportunity_names((search_state.get("answer") or "").strip()))
    turns = []
    for turn in search_state.get("discussion") or []:
        if turn["role"] == "user":
            turns.append("**You:** " + turn["content"].strip())
        else:
            model = f" (`{turn['model']}`)" if turn.get("model") else ""
            turns.append(f"**LLM**{model}:\n\n" + link_opportunity_names(turn["content"].strip()))
    if turns:
        parts.append("## Discussion\n\n" + "\n\n".join(turns))
    return "\n\n".join(parts)


def render_copy_buttons(buttons: list[tuple[str, str]], fit: bool = False) -> None:
    """A row of small copy buttons, one per (label, text), in a single component iframe.

    `fit` sizes the iframe to its buttons, so other elements can sit beside it on the same line.
    """
    buttons = [(label, text) for label, text in buttons if text and text.strip()]
    if not buttons:
        return
    payload = json.dumps([text for _label, text in buttons]).replace("</", "<\\/")
    labels = json.dumps([label for label, _text in buttons]).replace("</", "<\\/")
    # About 7px a character at 12px, plus the icon, padding and border; the body's side margins add 16px.
    # The row fills the frame's height with the buttons centred, so they aren't cut off at the bottom.
    width = sum(len(label) * 7 + 48 for label, _text in buttons) + 6 * (len(buttons) - 1) + 16 if fit else None
    components.html(
        f"""
<style>body{{margin:0 8px}}</style>
<div id="row" style="display:flex;gap:6px;height:32px;align-items:center"></div>
<script>
const texts = {payload};
const labels = {labels};
const row = document.getElementById("row");
labels.forEach((label, i) => {{
  const button = document.createElement("button");
  button.style.cssText = "font:12px -apple-system,BlinkMacSystemFont,sans-serif;color:#888;" +
    "background:transparent;border:1px solid rgba(136,136,136,.45);border-radius:6px;padding:2px 10px;cursor:pointer";
  button.textContent = "📋 " + label;
  button.addEventListener("click", async () => {{
    try {{
      await navigator.clipboard.writeText(texts[i]);
    }} catch (err) {{
      const area = document.createElement("textarea");
      area.value = texts[i];
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }}
    button.textContent = "✓ Copied";
    setTimeout(() => (button.textContent = "📋 " + label), 1500);
  }});
  row.appendChild(button);
}});
</script>
""",
        width=width,
        height=32,
    )


def install_filter_commit_on_blur() -> None:
    """Turn text typed in a filter field into a capsule when focus leaves it.

    Streamlit's multiselect discards typed-but-uncommitted text on blur. A capture-phase
    focusout listener on the page sends Enter first, so the term is added instead of lost.
    Only blurs the user caused (click elsewhere, Tab) commit; a rerun redrawing the page
    mid-typing would otherwise commit each letter as its own term.
    """
    components.html(
        """
<script>
const page = window.parent;
if (page.__filterCommitOnBlurVersion !== 2) {
  page.__filterCommitOnBlurVersion = 2;
  const boxOf = (el) => (el instanceof page.Element ? el.closest('[data-testid="stMultiSelect"]') : null);
  let userLeftAt = 0;
  page.document.addEventListener("pointerdown", (event) => {
    const box = boxOf(page.document.activeElement);
    if (box && !box.contains(event.target)) userLeftAt = Date.now();
  }, true);
  page.document.addEventListener("keydown", (event) => {
    if (event.key === "Tab" && boxOf(event.target)) userLeftAt = Date.now();
  }, true);
  page.document.addEventListener("focusout", (event) => {
    const input = event.target;
    if (!(input instanceof page.HTMLInputElement) || !input.value.trim()) return;
    const box = boxOf(input);
    if (!box || (event.relatedTarget && box.contains(event.relatedTarget))) return;
    if (Date.now() - userLeftAt > 500) return;
    const enter = { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true };
    input.dispatchEvent(new page.KeyboardEvent("keydown", enter));
  }, true);
}
</script>
""",
        height=0,
    )


# Text boxes that save when you leave them (their on_change saves), by widget key or key prefix.
SAVE_ON_BLUR_KEYS = (
    "corpus_prompt_", "corpus_root_path_input", "max_results", "connector_config_path_input",
    "exporter_config_path_input", "outbox_path_input", "question_text", "executor_title", "executor_url",
    "link_title_", "link_url_", "profile_short_name", "profile_text", "prompt_command_", "prompt_text_",
    "prompt_name_", "prompt_step_text_", "prompt_step_name_", "quick_search_term_", "reference_title_",
    "title_marker_text",
)
NOT_SAVED_WHEN_EMPTY = ("prompt_name_", "prompt_text_", "prompt_command_")  # an empty one keeps the old value


def install_save_flash() -> None:
    """Flash a text box green with a 'Saved' tag when leaving it (or Enter) saved a change."""
    keys = json.dumps(SAVE_ON_BLUR_KEYS)
    keep = json.dumps(NOT_SAVED_WHEN_EMPTY)
    components.html(
        f"""
<script>
const page = window.parent;
const doc = page.document;
if (page.__saveFlashVersion !== 1) {{
  page.__saveFlashVersion = 1;
  const keys = {keys}, keepIfEmpty = {keep}, green = "33, 195, 84";
  const style = doc.createElement("style");
  style.textContent = ".save-flash {{ position: absolute; pointer-events: none; z-index: 1000; border-radius: 0.5rem; "
    + `border: 2px solid rgb(${{green}}); background: rgba(${{green}}, 0.18); transition: opacity 0.6s ease; }}`
    + ".save-flash span {{ position: absolute; right: 6px; top: -11px; font: 600 11px -apple-system, sans-serif; "
    + `color: #fff; background: rgb(${{green}}); border-radius: 999px; padding: 1px 8px; }}`;
  doc.head.appendChild(style);
  // The saving key a field belongs to (e.g. "prompt_text_"), or null for boxes that don't save.
  const savingKey = (el) => {{
    const box = el instanceof page.Element ? el.closest('[class*="st-key-"]') : null;
    const key = box && [...box.classList].find((c) => c.startsWith("st-key-"));
    return key ? keys.find((k) => key.slice(7).startsWith(k)) || null : null;
  }};
  const isField = (el) => el instanceof page.HTMLTextAreaElement
    || (el instanceof page.HTMLInputElement && ["text", "number", ""].includes(el.type));
  const flash = (el) => {{
    const field = el.closest('[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"], '
      + '[data-testid="stNumberInputContainer"]') || el;
    const r = field.getBoundingClientRect();
    const mark = doc.createElement("div");
    mark.className = "save-flash";
    mark.innerHTML = "<span>✓ Saved</span>";
    Object.assign(mark.style, {{ left: `${{r.left + page.scrollX - 2}}px`, top: `${{r.top + page.scrollY - 2}}px`,
                                 width: `${{r.width + 4}}px`, height: `${{r.height + 4}}px` }});
    doc.body.appendChild(mark);
    setTimeout(() => (mark.style.opacity = "0"), 1100);
    setTimeout(() => mark.remove(), 1800);
  }};
  const before = new page.WeakMap();
  const check = (el) => {{
    const key = savingKey(el);
    if (!key || !before.has(el) || before.get(el) === el.value) return;
    before.set(el, el.value);
    if (!el.value.trim() && keepIfEmpty.includes(key)) return;  // not saved
    flash(el);
  }};
  doc.addEventListener("focusin", (event) => {{
    if (isField(event.target)) before.set(event.target, event.target.value);
  }}, true);
  doc.addEventListener("focusout", (event) => {{ if (isField(event.target)) check(event.target); }}, true);
  doc.addEventListener("keydown", (event) => {{
    if (event.key === "Enter" && event.target instanceof page.HTMLInputElement && isField(event.target)) {{
      setTimeout(() => check(event.target), 0);
    }}
  }}, true);
}}
</script>
""",
        height=0,
    )


def install_tooltip_hide_on_click() -> None:
    """Show a tooltip only while the pointer is on its element, and hide a button's tooltip once it is clicked.

    When a click reruns the app, Streamlit can leave tooltips open for good, even reopening those of buttons
    clicked earlier (e.g. an Agent prompt's full text over the page). Each tooltip's element points to it with
    aria-describedby, so a tooltip shows only while the pointer is on that element and it hasn't just been
    clicked.
    """
    components.html(
        """
<script>
const page = window.parent;
const doc = page.document;
if (page.__tooltipTidyVersion !== 3) {
  page.__tooltipTidyVersion = 3;
  const style = doc.createElement("style");
  style.textContent = "body > div.tip-hidden { visibility: hidden !important; }";
  doc.head.appendChild(style);
  let tipIds = 0;
  // An element with a tooltip, named by its widget key class (e.g. "st-key-use_agent_prompt_Recent_…"), which
  // stays the same when a rerun redraws it, else by an id given to the element.
  const nameOf = (el) => {
    const hover = el instanceof page.Element ? el.closest('[data-testid="stTooltipHoverTarget"]') : null;
    if (!hover) return null;
    const box = hover.closest('[class*="st-key-"]');
    const key = box && [...box.classList].find((c) => c.startsWith("st-key-"));
    return key || (hover.dataset.tipId ||= `tip-${++tipIds}`);
  };
  let pointerAt = null;  // the element under the pointer
  let clicked = null;  // the name of the element clicked last, until the pointer leaves it
  const update = () => {
    for (const layer of doc.body.children) {
      const tip = layer.tagName === "DIV" ? layer.querySelector('[role="tooltip"][id]') : null;
      if (!tip || !tip.querySelector('[data-testid="stTooltipContent"]')) {
        layer.classList.remove("tip-hidden");
        continue;
      }
      const owner = doc.querySelector(`[aria-describedby="${page.CSS.escape(tip.id)}"]`);
      const show = owner && pointerAt && owner.contains(pointerAt) && nameOf(owner) !== clicked;
      layer.classList.toggle("tip-hidden", !show);
    }
  };
  new page.MutationObserver(update).observe(doc.body, { childList: true, subtree: true, attributes: true,
                                                         attributeFilter: ["aria-describedby"] });
  doc.addEventListener("pointerdown", (event) => {
    clicked = nameOf(event.target);
    update();
  }, true);
  doc.addEventListener("pointermove", (event) => {
    const before = nameOf(pointerAt);
    pointerAt = event.target;
    const now = nameOf(pointerAt);
    if (now === before) return;
    clicked = null;
    update();
  }, true);
}
</script>
""",
        height=0,
    )


QUESTION_LABEL = "What would you like to ask?"
QUESTION_BIG_HEIGHT = 220
# The question box at one line's height (a text area keeps Enter for new lines and stays selectable on focus).
QUESTION_ONE_LINE_STYLE = """<style>
.st-key-question_text [data-testid="stTextAreaRootElement"] { height: auto !important; min-height: 0 !important; }
.st-key-question_text textarea { height: 2.4rem !important; min-height: 2.4rem !important; resize: none; }
</style>"""


def install_select_question_on_focus() -> None:
    """Select the whole question when the box is clicked into, so typing replaces it."""
    components.html(
        f"""
<script>
const page = window.parent;
if (!page.__selectQuestionOnFocus) {{
  page.__selectQuestionOnFocus = true;
  const label = {json.dumps(QUESTION_LABEL)};
  const isQuestion = (el) => el instanceof page.HTMLTextAreaElement && el.getAttribute("aria-label") === label;
  page.document.addEventListener("focusin", (event) => {{
    if (!isQuestion(event.target)) return;
    event.target.select();
    // The click that focused the box would otherwise place the caret on mouseup.
    event.target.__keepSelection = true;
    setTimeout(() => (event.target.__keepSelection = false), 400);
  }}, true);
  page.document.addEventListener("mouseup", (event) => {{
    if (isQuestion(event.target) && event.target.__keepSelection) {{
      event.preventDefault();
      event.target.__keepSelection = false;
    }}
  }}, true);
}}
</script>
""",
        height=0,
    )


def render_follow_up_discussion(
    search_state: dict,
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str = "",
    model_options: list[str] | None = None,
    safe_lib: str = "",
) -> tuple:
    """Chat panel for follow-up questions about the notes from the last search; returns the follow-up model and
    the dock holding the follow-up box (see render_discussion_body)."""
    st.subheader("Discussion")

    # Follow-ups can use a different (e.g. larger) model than the main answer.
    options = list(model_options or [model_name])
    saved_model = get_corpus_config(safe_lib).get("follow_up_model") or model_name
    if saved_model not in options:
        saved_model = model_name if model_name in options else options[0]
    # A prompt with its own model sets the follow-up model once per search; changing it afterwards overrides it.
    prompt_model = search_state.get("prompt_model")
    if prompt_model in options and not search_state.get("prompt_model_applied"):
        st.session_state[f"follow_up_model_{safe_lib}"] = prompt_model
        search_state["prompt_model_applied"] = True
    model_key = f"follow_up_model_{safe_lib}"
    if st.session_state.get(model_key) not in options:
        st.session_state[model_key] = saved_model  # set here, not via index, so a prompt's model can replace it
    # The picker is drawn in the Ask a follow-up box below; its value is already known from its key.
    model_name = st.session_state[model_key]
    in_use = len({discussion_note_key(r) for r in discussion_results(search_state)})
    total = len({discussion_note_key(r) for r in search_state["results"]})
    notes = "the notes in this search" if in_use == total else f"{in_use} of {total} notes from this search"
    caption = f"Follow-ups are answered from {notes} using `{model_name}`"
    st.caption(caption + describe_prompt_parts(search_state))
    # The task the model was given (for an Agent: its collection prompt plus how to list and where it's
    # heading), ready to paste into a flow or a test.
    render_copy_buttons([("Copy task prompt", search_state.get("task_prompt", ""))])

    # Sized to their labels, so a narrow panel wraps the row instead of cutting the labels short.
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        expanded = st.toggle(
            "Expand",
            key="discussion_expanded",
            help="Make the discussion panel much taller. For very long conversations, use **Pop out**.",
        )
        if st.button("⤢ Pop out", help="Open the discussion in a large window."):
            show_discussion_dialog(search_state, model_name, max_tokens, temperature, corpus_prompt)
        st.button(
            "⤢ Last reply on big screen",
            key="big_screen_last",
            disabled=not any(turn["role"] == "assistant" for turn in search_state["discussion"]),
            on_click=set_big_screen,
            args=(search_state, "last"),
            help="Show the latest reply in the main output area; it keeps following new replies.",
        )

    dock = render_discussion_body(
        search_state,
        model_name,
        max_tokens,
        temperature,
        corpus_prompt,
        key_prefix="panel",
        # A content-sized panel can be shorter than the default box, so expanding uses a fixed taller height.
        height=DISCUSSION_EXPANDED_HEIGHT if expanded else DISCUSSION_HEIGHT,
        model_select=(model_key, options, safe_lib),
    )
    return model_name, dock


def sticky_follow_up_style(box_key: str) -> str:
    """CSS that keeps the follow-up box and the Agent box (the dock holding both) in view while a long answer
    beside them is scrolled, so you can answer its questions without scrolling back.

    A sticky element only travels within its parent, so the Discussion column is stretched to the full
    height of the answer column first. The dock gets the page background so text scrolling under it is hidden,
    and scrolls within itself when it's taller than the window.
    """
    background = "#0e1117" if st.context.theme.type == "dark" else "#ffffff"
    # A column holds one .stVerticalBlock; the keyed box (or the wrapper carrying its key) sits directly in it.
    column = f'[data-testid="stColumn"]:has(> .stVerticalBlock > .st-key-{box_key})'
    return f"""<style>
[data-testid="stHorizontalBlock"]:has({column}) {{ align-items: stretch; }}
{column} {{ align-self: stretch; }}
{column} > .stVerticalBlock {{ height: 100%; }}
.st-key-{box_key} {{ position: sticky; top: 4.25rem; z-index: 5; background: {background};
  max-height: calc(100vh - 5.5rem); overflow-y: auto; overflow-x: hidden; }}
</style>"""


def render_discussion_body(
    search_state: dict,
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str,
    *,
    key_prefix: str,
    height: int | str,
    in_dialog: bool = False,
    model_select: tuple[str, list[str], str] | None = None,
):
    """Transcript, follow-up box and clear button, shared by the side panel and the pop-out. Returns the dock
    holding the follow-up box, for the Agent box to go in too.

    `model_select` (widget key, model options, library key) puts the Follow-up model picker in the box."""
    discussion = search_state["discussion"]
    with st.container(height=height, border=True, autoscroll=True):
        if not discussion:
            st.caption("No follow-up questions yet.")
        for index, turn in enumerate(discussion):
            with st.chat_message(turn["role"]):
                st.markdown(link_opportunity_names(turn["content"]))
                if turn.get("model"):
                    st.caption(f"`{turn['model']}`")
                # Every turn can be copied. When the model was sent more than is shown (a step's prompt, or
                # Needs work with the rejected list), that full prompt can be copied too, to reuse in flows.
                buttons = [("Copy", link_opportunity_names(turn["content"]) if turn["role"] == "assistant"
                             else turn["content"])]
                if turn.get("prompt") and turn["prompt"] != turn["content"]:
                    buttons.append(("Copy prompt sent", turn["prompt"]))
                # Copy and Big screen share one line.
                with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                    render_copy_buttons(buttons, fit=True)
                    if turn["role"] == "assistant":
                        on_big_screen = big_screen_turn(search_state)
                        if on_big_screen and on_big_screen[0] == index:
                            st.caption("Showing on the big screen", width="content")
                        elif in_dialog:
                            # A full rerun closes the pop-out so the reply can be read in the main area.
                            if st.button("⤢ Big screen", key=f"{key_prefix}_big_screen_{index}", type="tertiary"):
                                set_big_screen(search_state, index)
                                st.rerun()
                        else:
                            st.button(
                                "⤢ Big screen",
                                key=f"{key_prefix}_big_screen_{index}",
                                type="tertiary",
                                on_click=set_big_screen,
                                args=(search_state, index),
                                help="Show this reply in the main output area.",
                            )

    render_follow_up_context(search_state, key_prefix)
    if not in_dialog:
        st.html(sticky_follow_up_style(f"{key_prefix}_follow_up_dock"))
    # The side panel's dock: the follow-up box, and the Agent box the caller adds to it, stay in view together.
    dock = st.container() if in_dialog else st.container(key=f"{key_prefix}_follow_up_dock")
    with dock, st.container(border=True, key=f"{key_prefix}_follow_up_box"):
        # The toggle sits outside the form: a widget inside a form only takes effect when the form is sent.
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("Ask a follow-up", width="stretch")
            if model_select:
                select_key, options, safe_lib = model_select
                model_name = st.selectbox(
                    "Follow-up model",
                    options=options,
                    key=select_key,
                    on_change=persist_follow_up_model,
                    args=(safe_lib,),
                    label_visibility="collapsed",
                    width=190,
                    help=(
                        "Ollama model for the next follow-up. It can differ from the main answer model: a larger "
                        "one for deeper discussion, a smaller one for a quick reply. Saved per corpus."
                    ),
                )
            big_box = st.toggle(
                "Expand",
                key=f"{key_prefix}_follow_up_expanded",
                help="Make the follow-up box bigger for longer replies.",
            )
        if model_select:
            st.caption(describe_llm_speed(model_name))
        with st.form(f"{key_prefix}_follow_up_form", clear_on_submit=True, border=False):
            follow_up = st.text_area(
                "Ask a follow-up",
                key=f"{key_prefix}_follow_up_text",  # a fixed key keeps typed text when the box resizes
                height=FOLLOW_UP_BOX_EXPANDED_HEIGHT if big_box else FOLLOW_UP_BOX_HEIGHT,
                placeholder="e.g. Who owns each of those requirements?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

    # Inside the pop-out, rerun only the dialog so it stays open.
    rerun_scope = "fragment" if in_dialog else "app"

    if discussion and st.button("Clear discussion", key=f"{key_prefix}_clear", use_container_width=True):
        discussion.clear()
        search_state["big_screen"] = None
        st.rerun(scope=rerun_scope)

    if submitted and follow_up.strip():
        if add_discussion_turn(search_state, follow_up.strip(), model_name, max_tokens, temperature, corpus_prompt):
            st.rerun(scope=rerun_scope)
    return dock


@st.dialog("Discussion", width="large", on_dismiss="rerun")
def show_discussion_dialog(
    search_state: dict,
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str,
) -> None:
    """Pop-out window for long conversations; closing it refreshes the side panel."""
    st.caption(f"Using `{model_name}`" + describe_prompt_parts(search_state))
    render_discussion_body(
        search_state,
        model_name,
        max_tokens,
        temperature,
        corpus_prompt,
        key_prefix="dialog",
        height=600,
        in_dialog=True,
    )


def render_library_counts(search_state: dict, key_prefix: str = "results") -> None:
    """Per-library match counts for all-library searches, with buttons to switch library.

    `key_prefix` keeps button keys unique when the gathered notes and an answer both show counts.
    """
    library_counts = search_state.get("library_counts") or []
    results = search_state["results"]
    if library_counts:
        shown_notes = len({(r.get("_library"), r.get("file_source")) for r in results})
        total_notes = sum(count for _library, count in library_counts)
        with st.expander(
            f"Matches by library — {total_notes} note(s) in {len(library_counts)} library(ies)",
            expanded=True,
        ):
            if shown_notes < total_notes:
                st.caption(f"Showing {shown_notes} of {total_notes} notes — raise **Max results** to see more.")
            for library, count in library_counts:
                name_col, button_col = st.columns([3, 1], vertical_alignment="center")
                name_col.markdown(f"**{library}** — {count} note(s)")
                button_col.button(
                    "Use library",
                    key=f"use_library_{key_prefix}_{library}",
                    on_click=switch_to_library,
                    args=(library,),
                    disabled=library == st.session_state.get("selected_library"),
                    use_container_width=True,
                    help="Switch to this library, e.g. to run a semantic search inside it.",
                )


FOLD_FOLDERS_OVER = 50  # in-scope notes beyond which folders fold up, so a long listing stays quick
OPEN_FOLDERS_KEY = "gathered_open_folders"  # ids of the folders opened with ▸ while folded
FOLDER_SELECTION_HELP = ("🚫 excludes every note in this folder and its sub-folders. 🎯 keeps them and excludes the "
                         "other listed notes you haven't picked with 🎯.")


def toggle_open_folder(folder_id: str) -> None:
    opened = st.session_state.setdefault(OPEN_FOLDERS_KEY, set())
    (opened.discard if folder_id in opened else opened.add)(folder_id)


def folder_tree(notes: dict, safe_lib: str) -> list[dict]:
    """The listed notes' folders, parents before children: {"path", "label", "depth", "direct": the notes in the
    folder itself, "notes": (library key, file) of every note under it}. A folder holding no notes of its own
    and a single sub-folder is merged with it ("Clients > Customers"); notes whose folder isn't known come last."""
    root: dict = {"children": {}, "direct": []}
    for note_key, blocks in notes.items():
        path = blocks[0].get("_folder_path", "")
        node = root
        for part in path.split(FOLDER_PATH_SEPARATOR) if path else ["\uffff"]:  # sorts last
            node = node["children"].setdefault(part, {"children": {}, "direct": []})
        node["direct"].append((note_key, blocks))

    def under(node: dict) -> list[tuple[str, str]]:
        own = [(blocks[0].get("_safe_lib", safe_lib), file_source) for (_lib, file_source), blocks in node["direct"]]
        return own + [key for child in node["children"].values() for key in under(child)]

    nodes: list[dict] = []

    def walk(node: dict, path: list[str], depth: int) -> None:
        for name in sorted(node["children"], key=str.lower):
            child, parts = node["children"][name], [name]
            while not child["direct"] and len(child["children"]) == 1:
                (next_name, child), = child["children"].items()
                parts.append(next_name)
            label = "Folder not known" if parts == ["\uffff"] else FOLDER_PATH_SEPARATOR.join(parts)
            nodes.append({"path": path + parts, "label": label, "depth": depth, "direct": child["direct"],
                          "notes": under(child)})
            walk(child, path + parts, depth + 1)

    walk(root, [], 0)
    return nodes


def render_gathered_notes(gathered: dict, library_folder: Path, safe_lib: str, pick: dict | None = None) -> None:
    """The notes gathered by Find notes and Filter search only, one row per note, most recent hit first.

    Each row has the colour of the route that found the note last: blue semantic, orange literal.
    """
    keys = list(reversed(gathered["notes"]))
    results = [block for key in keys for block in gathered["notes"][key]["blocks"]]
    routes = {key: gathered["notes"][key]["route"] for key in keys}
    render_library_counts({"library_counts": gathered["library_counts"], "results": results}, key_prefix="gathered")

    notes: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        notes.setdefault((result.get("_library", ""), str(result.get("file_source", ""))), []).append(result)
    by_route = {route: sum(1 for r in routes.values() if r == route) for route in ROUTE_TINTS}
    kept = kept_gathered_notes(gathered)
    st.markdown(
        f"**In scope · {len(notes)} note(s)** · {by_route['semantic']} by meaning (blue), "
        f"{by_route['literal']} by exact text (orange)"
    )
    st.caption(
        "Every search adds the notes it finds here; a note found again takes the colour of the latest search. "
        "Tick **🚫** to exclude a note or click **🎯** to keep just the ones you pick, then **Ask about in-scope "
        f"notes** answers from the {len(kept)} you kept. **Clear results** starts over."
    )
    listed = [
        (blocks[0].get("_safe_lib", safe_lib), file_source)
        for (_library, file_source), blocks in notes.items()
    ]
    # Select-all box: ticked only when every listed note is excluded.
    st.session_state["exclude_all_notes"] = all(
        file_source in get_excluded_notes(lib_key) for lib_key, file_source in listed
    )
    check_col, target_col, help_col, label_col = st.columns([1, 1, 0.6, 9.4], vertical_alignment="center")
    with check_col:
        st.checkbox("🚫", key="exclude_all_notes", on_change=toggle_all_notes_excluded, args=(listed,))
    target_col.markdown("🎯")
    help_col.markdown("\u200b", help=NOTE_SELECTION_HELP)
    label_col.caption("Exclude all")
    st.markdown('<hr style="margin:0.1rem 0 0.6rem">', unsafe_allow_html=True)

    # Grouped under their folders as a tree, each folder with 🚫 and 🎯 for all the notes under it. Past
    # FOLD_FOLDERS_OVER notes the folders fold up and a folder's notes are drawn only when it's opened.
    folded = len(notes) > FOLD_FOLDERS_OVER
    if folded:
        st.caption(f"Over {FOLD_FOLDERS_OVER} notes, so folders are folded: **▸** opens one. 🚫 and 🎯 on a folder "
                   "act on every note in it and its sub-folders.")
    opened = st.session_state.setdefault(OPEN_FOLDERS_KEY, set())
    row = 0
    for node in folder_tree(notes, safe_lib):
        folder_id = hashlib.md5(json.dumps(node["path"]).encode()).hexdigest()[:12]
        under = node["notes"]
        excluded = sum(1 for lib, source in under if source in get_excluded_notes(lib))
        box = f"exclude_folder_{folder_id}"
        st.session_state[box] = excluded == len(under)
        indent = "\u2003" * node["depth"]
        text = (f"{indent}📁 **{node['label']}** · {plural(len(under), 'note')}"
                + (f" · {excluded} excluded" if excluded else ""))
        check_col, target_col, help_col, label_col = st.columns([1, 1, 0.6, 9.4], vertical_alignment="center")
        check_col.checkbox("🚫", key=box, on_change=toggle_folder_excluded, args=(box, under))
        target_col.button("🎯", key=f"target_folder_{folder_id}", type="tertiary", on_click=target_notes,
                          args=(under, listed))
        help_col.markdown("\u200b", help=FOLDER_SELECTION_HELP)
        is_open = not folded or folder_id in opened
        if folded:
            label_col.button(f"{'▾' if is_open else '▸'} {text}", key=f"open_folder_{folder_id}", type="tertiary",
                             on_click=toggle_open_folder, args=(folder_id,),
                             help="Show or hide the notes directly in this folder")
        else:
            label_col.markdown(text)
        if not is_open:
            continue
        for (library, file_source), blocks in node["direct"]:
            route = routes[(blocks[0].get("_safe_lib", safe_lib), file_source)]
            with st.container(key=f"gathered_{route}_{row}"):
                render_filter_result_row(library, file_source, blocks, listed, library_folder, safe_lib,
                                         {"uploads_dirs": ()}, pick)
            row += 1
    if pick is not None:
        render_pick_done(pick, "pick_done_bottom")


def render_filter_result_row(
    library: str, file_source: str, blocks: list, listed: list, library_folder: Path, safe_lib: str,
    search_state: dict, pick: dict | None = None,
) -> None:
    """One note in the Filter search list: exclude box, target button and an expander with the note."""
    note_lib = blocks[0].get("_safe_lib", safe_lib)
    excluded = file_source in get_excluded_notes(note_lib)
    name = Path(file_source).name
    label = (f"~~{name}~~ · excluded" if excluded else name) + (
        f"  ·  {library}" if library and not blocks[0].get("_folder_path") else ""
    ) + action_badges(note_lib, file_source)
    if pick is not None:  # pick mode: add this note to the reference document, or take it out
        found = resolve_source_path(file_source, blocks[0].get("_library_folder", library_folder),
                                    blocks[0].get("_uploads_dirs", search_state["uploads_dirs"]))
        if found is not None:
            note = reference_note_for_path(found)
            ref = find_reference_document(pick["id"]) or pick
            added = any(reference_note_key(n) == reference_note_key(note) for n in ref.get("notes", []))
            with st.container(horizontal=True, gap="small"):
                st.button(
                    f"✓ In {reference_label(ref)} — remove" if added else f"Add to {reference_label(ref)}",
                    key=f"pick_toggle_{note_lib}_{file_source}",
                    type="primary" if not added else "secondary",
                    icon=":material/remove:" if added else ":material/add:",
                    on_click=toggle_reference_note,
                    args=(ref["id"], note),
                )
                st.button("Open", key=f"pick_open_{note_lib}_{file_source}", icon=":material/visibility:",
                          on_click=open_document_popup, args=(note,),
                          help="Read the note in a popup; the list and your filters stay as they are.")
    check_col, target_col, help_col, note_col = st.columns([1, 1, 0.6, 9.4])
    with check_col:
        checkbox_key = exclude_checkbox_key(note_lib, file_source)
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = excluded
        st.checkbox(
            "🚫",
            key=checkbox_key,
            on_change=toggle_excluded_note,
            args=(note_lib, file_source),
        )
    with target_col:
        st.button(
            "🎯",
            key=f"target::{note_lib}::{file_source}",
            type="tertiary",
            on_click=target_note,
            args=(note_lib, file_source, listed),
        )
    help_col.markdown("\u200b", help=NOTE_SELECTION_HELP)
    with note_col:
        with st.expander(label):
            note_folder = blocks[0].get("_library_folder", library_folder)
            note_uploads = blocks[0].get("_uploads_dirs", search_state["uploads_dirs"])
            path = resolve_source_path(file_source, note_folder, note_uploads)
            st.caption(str(path) if path else f"Source: `{file_source}`")
            render_note_actions(note_lib, file_source)
            render_full_document(file_source, path, note_lib, "gathered")


RECEIPT_STATUS_TEXT = {
    "done": "everything was created",
    "partial": "some items were created; the rest need attention",
    "needs_review": "nothing was created yet; a record or field needs choosing",
    "failed": "it couldn't be processed",
}


def staged_item_title(item: dict) -> str:
    if item.get("type") == "update":
        return f"Next Step: {item.get('value', '')}"
    return item.get("subject") or item.get("title") or "(untitled)"


def describe_staged_item(item: dict) -> str:
    """One readable line (plus a short body) for a staged note, task, email or Next Step update."""
    if item["type"] == "update":
        name = item["related_to"]["name"] or "no opportunity yet"
        text = f"**Next Step** on *{name}*  \n→ {item['value']}"
        if len(item["value"]) > NEXT_STEP_LIMIT:
            text += f"  \n⚠ {len(item['value'])} characters: a Next Step holds {NEXT_STEP_LIMIT}, so the connector will hold it."
        if "previous" in item:
            text += f"  \nNow: {item['previous'] or '(empty)'}"
        return text
    details = []
    if item["type"] == "task":
        details += [
            f"due {item['due_date'] or 'not set'}",
            f"priority {item['priority'] or 'not set'}",
            f"owner {item['owner'] or 'not given'}",
        ]
    elif item["type"] == "email":
        details.append("to " + (", ".join(item["to"]) or "nobody given"))
    related = item["related_to"]
    if related["name"]:
        details.append(f"related to {related['name']}" + (f" ({related['kind']})" if related["kind"] else ""))
    text = f"**{staged_item_title(item)}** — " + " · ".join(details) if details else f"**{staged_item_title(item)}**"
    body = item.get("description") or item.get("body") or ""
    return text + (f"  \n{truncate_text(body, 200)}" if body else "")


STAGED_FIELD_NAMES = {
    "subject": "Subject", "title": "Title", "description": "Description", "body": "Body", "to": "To",
    "due_date": "Due date", "priority": "Priority", "owner": "Owner", "related_to": "Related to",
    "field": "Field", "value": "New value", "previous": "Current value", "allow_duplicate": "Create anyway",
}
LONG_STAGED_FIELDS = ("description", "body", "value", "previous")  # shown in full in a copyable box


def render_full_staged_item(item: dict) -> None:
    """Folded under a staged item: every field exactly as it goes into the package, long text in full."""
    sent = {key: value for key, value in item.items() if not key.startswith("_") and key != "type"}
    with st.expander("Everything that will be sent", icon=":material/visibility:"):
        for key, value in sent.items():
            name = STAGED_FIELD_NAMES.get(key, key.replace("_", " ").capitalize())
            if key in LONG_STAGED_FIELDS:
                st.caption(f"**{name}**" + ("" if value else " · empty"))
                if value:
                    st.code(str(value), language=None, wrap_lines=True)
                continue
            if key == "related_to":
                value = (value or {}).get("name", "") + (f" ({value['kind']})" if (value or {}).get("kind") else "")
            elif key == "to":
                value = ", ".join(value or [])
            elif isinstance(value, bool):
                value = "yes" if value else "no"
            st.markdown(f"**{name}:** {value if value not in ('', None) else '*not set*'}")


def unstage_item(agent: dict, index: int) -> None:
    agent["staged"]["items"].pop(index)
    # Item positions shift, so drop the record pickers' remembered choices for this staging.
    prefixes = tuple(f"{p}_{agent['staged']['id']}_" for p in ("agent_ref", "agent_dup", "agent_ns_replace"))
    for key in [k for k in st.session_state if str(k).startswith(prefixes)]:
        del st.session_state[key]


KEEP_RELATED = "__keep__"
NO_RELATED = "__none__"


def set_staged_related(agent: dict, index: int, widget_key: str) -> None:
    """Apply the record picked for a staged item: a reference opportunity, or no record."""
    choice = st.session_state.get(widget_key)
    item = agent["staged"]["items"][index]
    if choice == NO_RELATED:
        item["related_to"] = {"name": "", "kind": ""}
    elif choice and choice != KEEP_RELATED:
        item["related_to"] = {"name": choice, "kind": "opportunity"}
    if item["type"] == "update":
        # The connector only overwrites the Next Step it was told the user saw, so keep that in step.
        record = find_reference_record(item["related_to"]["name"])
        if record is not None:
            item["previous"] = record.get("next_step", "")
        else:
            item.pop("previous", None)
        item.pop("_already_said", None)  # it was about the opportunity picked before


def render_reference_picker(agent: dict, index: int, item: dict) -> None:
    """Under a staged item: whether its Related to is a real open opportunity, and a picker to choose one."""
    records = reference_records()
    if not records:
        return
    related = item["related_to"]
    match = find_reference_record(related["name"]) if related["kind"] in ("", "opportunity") else None
    if match:
        link = reference_record_link(match)
        st.caption(f"✓ Opportunity *{match['name']}* · {describe_reference_record(match)}" + (f" · {link}" if link else ""))
    elif related["name"] and related["kind"] in ("", "opportunity"):
        st.caption(f"⚠ *{related['name']}* isn't an open opportunity in the reference: choose one below, or the "
                   "connector will hold it.")
    elif related["name"]:
        st.caption(f"Related to {related['kind']} *{related['name']}* (only opportunities are in the reference).")
    elif item["type"] in ("note", "update"):
        st.caption(f"⚠ {'A note' if item['type'] == 'note' else 'A Next Step'} must be on a record: "
                   "choose its opportunity below.")
    names = [r["name"] for r in records]
    must_link = item["type"] in ("note", "update")  # notes and Next Step updates need a record; tasks don't
    options = ([] if match else [KEEP_RELATED]) + names + ([] if must_link else [NO_RELATED])
    by_name = {r["name"]: r for r in records}

    def label(option: str) -> str:
        if option == KEEP_RELATED:
            return f"Keep “{related['name']}”" if related["name"] else "Choose an opportunity…"
        if option == NO_RELATED:
            return "No record (standalone)"
        return f"{option} — {describe_reference_record(by_name[option])}"

    widget_key = f"agent_ref_{agent['staged']['id']}_{index}"
    st.selectbox(
        "Opportunity",
        options,
        index=options.index(match["name"]) if match else 0,
        format_func=label,
        key=widget_key,
        on_change=set_staged_related,
        args=(agent, index, widget_key),
        label_visibility="collapsed",
        help="Open opportunities from the connector's reference snapshot. The connector links to the one you choose.",
    )


def set_allow_duplicate(agent: dict, index: int, widget_key: str) -> None:
    item = agent["staged"]["items"][index]
    if st.session_state.get(widget_key):
        item["allow_duplicate"] = True
    else:
        item.pop("allow_duplicate", None)


NEXT_STEP_SIGNATURE = re.compile(r"\d{4}-\d{2}-\d{2}(-[A-Za-z0-9]+)?:\s*")  # "2026-10-25-JDOE: "


def same_next_step(new_value: str, current: str) -> bool:
    """True when a Next Step says what the current one does, by key words, ignoring dates, initials and marker."""
    return same_task_action(NEXT_STEP_SIGNATURE.sub(" ", new_value), NEXT_STEP_SIGNATURE.sub(" ", current))


def replace_next_step_anyway(agent: dict, index: int) -> None:
    """The user saw the Next Step it changed to and still wants theirs: the connector checks against that now."""
    item = agent["staged"]["items"][index]
    record = find_reference_record(item["related_to"]["name"])
    if record is not None:
        item["previous"] = record.get("next_step", "")


DIFF_REMOVED = "background: rgba(255, 75, 75, 0.18); text-decoration: line-through; border-radius: 3px;"
DIFF_ADDED = "background: rgba(33, 195, 84, 0.22); border-radius: 3px;"


def diff_html(old: str, new: str) -> str:
    """Word-level diff of two field values: removed words struck through on red, added words on green."""
    before, after = re.findall(r"\S+\s*", old), re.findall(r"\S+\s*", new)
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if op == "equal":
            parts.append(html.escape("".join(after[j1:j2])))
            continue
        if op in ("delete", "replace"):
            parts.append(f'<span style="{DIFF_REMOVED}">{html.escape("".join(before[i1:i2]).rstrip())}</span> ')
        if op in ("insert", "replace"):
            parts.append(f'<span style="{DIFF_ADDED}">{html.escape("".join(after[j1:j2]))}</span>')
    return "".join(parts)


def render_update_diff(item: dict) -> None:
    """A staged Next Step as a diff: what the field says now (from the snapshot) and what it will say."""
    name = item["related_to"]["name"] or "no opportunity yet"
    st.markdown(link_opportunity_names(f"**Next Step** on *{name}*"))
    new = item["value"]
    row = 'style="display: flex; gap: 0.5rem;"'
    label = 'style="flex: 0 0 2.2rem; opacity: 0.6; font-size: 0.85em; padding-top: 0.1em;"'
    if "previous" in item:
        now = html.escape(item["previous"]) if item["previous"] else "<i>(empty)</i>"
        change = diff_html(item["previous"], new)
    else:  # no snapshot entry: it's replaced whatever it says
        now, change = "<i>not known (not in the snapshot)</i>", f'<span style="{DIFF_ADDED}">{html.escape(new)}</span>'
    st.html(
        f'<div style="line-height: 1.6;"><div {row}><span {label}>Now</span><span>{now}</span></div>'
        f'<div {row}><span {label}>New</span><span>{change}</span></div></div>'
    )
    if len(new) > NEXT_STEP_LIMIT:
        st.caption(f"⚠ {len(new)} characters: a Next Step holds {NEXT_STEP_LIMIT}, so the connector will hold it.")


def render_next_step_checks(agent: dict, index: int, item: dict) -> None:
    """Under a staged Next Step: whether it changed since it was staged, or already says this."""
    if item.get("type") != "update" or "previous" not in item:
        return
    record = find_reference_record(item["related_to"]["name"])
    if record is None:
        return
    current = record.get("next_step", "")
    if current != item["previous"]:
        st.caption(
            f"⚠ Changed since you staged this: the Next Step is now “{current or '(empty)'}”. The connector would "
            "hold yours as changed since staged, unless you choose **Replace it anyway**."
        )
        st.button(
            "Replace it anyway",
            key=f"agent_ns_replace_{agent['staged']['id']}_{index}",
            icon=":material/swap_horiz:",
            on_click=replace_next_step_anyway,
            args=(agent, index),
            help="Overwrite the Next Step it has now with yours.",
        )
    elif current and (item.get("_already_said") or same_next_step(item["value"], current)):
        st.caption(
            f"⚠ The current Next Step already says this: “{current}”. Remove it with ✕ (and its chase task, if you "
            "don't need one) unless you want to restate it."
        )


def render_duplicate_check(agent: dict, index: int, item: dict) -> None:
    """Under a staged task: the open tasks it looks like, and Create anyway (else the connector holds it)."""
    matches = similar_open_tasks(item)
    if not matches:
        return
    st.caption(
        "⚠ Looks like an open task of yours, so the connector will hold it unless you tick **Create anyway**:  \n"
        + "  \n".join(describe_open_task(task) for task in matches[:3])
    )
    widget_key = f"agent_dup_{agent['staged']['id']}_{index}"
    st.checkbox(
        "Create anyway",
        value=bool(item.get("allow_duplicate")),
        key=widget_key,
        on_change=set_allow_duplicate,
        args=(agent, index, widget_key),
        help="Create this task even though it looks like one you already have.",
    )


def start_needs_work(agent: dict) -> None:
    agent["rejected"], agent["staged"], agent["needs_work"] = agent["staged"], None, True


def send_staged_to_outbox(
    search_state: dict, prompts: list[dict], source_notes: list[tuple[str, Path | None]]
) -> None:
    agent = search_state["agent"]
    staged = agent["staged"]
    try:
        package = build_handoff_package(search_state, prompts, staged, source_notes)
        path = write_handoff_package(package, get_outbox_dir())
    except Exception as exc:
        agent["send_error"] = f"Couldn't write to the outbox: {exc}"
        return
    log_sent_actions(search_state, prompts, staged, path)
    agent["sent"].append({
        "file": str(path),
        "type": staged["type"],
        "items": staged["items"],
        "at": datetime.datetime.now().strftime("%H:%M"),
    })
    agent["staged"] = None


def render_sent_batch(batch: dict) -> bool:
    """One sent package: what happened to it, and a link to it on the Actions page.
    Returns True while it's still waiting for a connector."""
    waiting = render_sent_status(batch)
    package_name = Path(batch["file"]).name
    st.link_button(
        "View in outbox",
        f"{ACTIONS_PAGE_URL}&package={package_name}",
        icon=":material/outbox:",
        type="tertiary",
        help=f"Opens the Actions page in a new tab, showing {package_name}: what it holds and what happened to it.",
    )
    return waiting


def render_sent_status(batch: dict) -> bool:
    """What happened to one sent package; returns True while it's still waiting for a connector."""
    count = f"{len(batch['items'])} {OUTPUT_TYPES.get(batch['type'], 'items').lower()}"
    package_file = Path(batch["file"])
    if (package_file.parent / WITHDRAWN_DIR / package_file.name).is_file():
        st.info(f"Sent {count} at {batch['at']}, then withdrawn on the Actions page.")
        return False
    receipt = read_handoff_receipt(package_file)
    if not receipt and not package_file.is_file():  # not in the outbox, withdrawn/, processed/ or failed/
        st.info(f"Sent {count} at {batch['at']}, then withdrawn and deleted on the Actions page.")
        return False
    if not receipt:
        st.info(f"Sent {count} to the outbox at {batch['at']}. Waiting for a connector to pick it up.")
        return True
    message = (
        f"Sent {count} at {batch['at']}. **{receipt.get('connector', 'Connector').capitalize()}:** "
        f"{RECEIPT_STATUS_TEXT.get(receipt['status'], receipt['status'])}."
    )
    if receipt.get("error"):
        message += f" {receipt['error']}"
    (st.success if receipt["status"] == "done" else st.warning)(message)
    for result in receipt.get("items", []):
        index = result["index"]
        title = staged_item_title(batch["items"][index]) if index < len(batch["items"]) else f"Item {index + 1}"
        if result["status"] in DONE_ITEM_STATUSES:
            if links := receipt_links(result):  # done: just where to see it
                st.caption(f"{title}: " + " · ".join(links))
            continue
        candidates = ", ".join(describe_candidate(c) for c in result.get("candidates", []))
        label = "Looks like" if result["status"] == "duplicate" else "Possible matches"
        st.caption(f"{title}: {result['message']}" + (f" {label}: {candidates}." if candidates else ""))
    return False


def render_agent_panel(
    search_state: dict,
    source_notes: list[tuple[str, Path | None]],
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str,
) -> None:
    """Agent workflow under the discussion: step buttons, Agent:<name> to stage, That's right / Needs work."""
    agent = search_state["agent"]
    # Read the prompts live, so edits made in the prompt library apply on the next click. Searches from
    # before several Agents could be injected hold one "prompt".
    library = load_prompt_library()
    prompts = [find_prompt(library, p["id"]) or p for p in agent.get("prompts") or [agent["prompt"]]]
    several = len(prompts) > 1
    label = " + ".join(agent_prompt_label(prompt) for prompt in prompts)
    item_kind = " and ".join(OUTPUT_TYPES[get_prompt_output_type(prompt)].lower() for prompt in prompts)
    add_agent_border_style(".st-key-agent_panel", ".st-key-agent_run button", ".st-key-agent_staging")

    with st.container(border=True, key="agent_panel"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{label}**", width="content")
            for prompt in prompts:
                st.link_button(
                    f"Edit {prompt['name']}" if several else "Edit prompt workflow",
                    f"{PROMPT_LIBRARY_PAGE_URL}&prompt={prompt['id']}",
                    icon=":material/open_in_new:",
                    type="tertiary",
                    help="Opens this Agent's steps and Action Command in the prompt library, in a new tab.",
                )
        for prompt in prompts:
            steps = [s for s in prompt.get("steps", []) if s.get("text", "").strip()]
            if not steps:
                continue
            with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
                if several:
                    st.caption(f"{prompt['name']}:")
                for number, step in enumerate(steps, 2):
                    if st.button(f"{number} · {step['name']}", key=f"agent_step_{step['id']}", help=step["text"]):
                        if add_discussion_turn(
                            search_state, f"*{step['name']}* — {step['text']}", model_name, max_tokens,
                            temperature, corpus_prompt, sent_text=step["text"],
                        ):
                            st.rerun()
        # Guidance, folded so the step buttons and Execute stay close together.
        with st.expander("How it works", expanded=False):
            st.caption(
                "Reply in the discussion to pick items (e.g. *Yes:1,2,3 ignore 4,5,6*), set dates and answer "
                f"questions. When it's right, stage the {item_kind}. Untick notes under Sources to narrow what "
                "the LLM uses."
            )
            snapshot = load_reference_snapshot()
            if snapshot:
                refreshed = datetime.datetime.fromisoformat(snapshot["generated_at"]).strftime("%d %b %H:%M")
                st.caption(
                    f"Reference: {len(reference_records())} {snapshot.get('scope', 'records')} from "
                    f"{snapshot.get('source', 'the connector')} ({snapshot.get('target', '?')}), refreshed "
                    f"{refreshed}. The Agent uses their exact names; refresh it from the task executor."
                )
            else:
                st.caption(
                    "No reference snapshot yet: the task executor can refresh one so the Agent uses real record names."
                )
        # Folded until needed; its title says when a note is waiting to go with Execute.
        note_key = f"agent_last_note_{search_state.get('id', '')}"
        typed = str(st.session_state.get(note_key, "")).strip()
        with st.expander("One last note" + (" · will be sent" if typed else ""), icon=":material/edit_note:"):
            last_note = st.text_area(
                "One last note",
                key=note_key,
                height=90,
                placeholder="e.g. exclude points 2-5, or make them all due Friday",
                label_visibility="collapsed",
                help="Sent with **Execute** only, as a final instruction that overrides the discussion where they "
                "differ. Saves another round in the discussion before staging.",
            )
        if st.button(
            f"Execute :material/smart_toy: {label} with context",
            key="agent_run",
            use_container_width=True,
            help=(
                "**Action Command" + ("s" if several else "") + ":** "
                + "; ".join(agent_command(prompt) for prompt in prompts)
                + "\n\nStages what you've agreed so you can check it first"
                + (", asking the model once per Agent; it's all sent as one package." if several else ".")
            ),
        ):
            with st.spinner(f"`{model_name}` is staging the {item_kind}…"):
                try:
                    agent["staged"] = stage_agent_batch(
                        search_state, prompts, model_name, max_tokens, temperature, corpus_prompt, last_note
                    )
                    agent["needs_work"] = False
                except Exception as exc:
                    st.error(f"Couldn't stage the {item_kind}: {exc}")
                else:
                    st.rerun()

        staged = agent.get("staged")
        if staged:
            with st.container(border=True, key="agent_staging"):
                st.markdown(f"**Staged — check before it goes to the outbox**  \n{staged['summary']}")
                if any(i["type"] == "update" for i in staged["items"]) and not profile_short_name(load_prompt_library()):
                    st.warning("Next Steps are signed with the date only: set your **Short name** in the prompt "
                               "library's **About me** (e.g. JDOE), then stage again.")
                if not staged["items"]:
                    st.info("Nothing to send: no agreed items were found. Choose **Needs work** and say what you want.")
                for index, item in enumerate(staged["items"]):
                    item_col, remove_col = st.columns([0.92, 0.08])
                    with item_col:
                        if item["type"] == "update":
                            render_update_diff(item)
                        else:
                            st.markdown(link_opportunity_names(describe_staged_item(item)))
                        render_full_staged_item(item)
                        render_reference_picker(agent, index, item)
                        render_duplicate_check(agent, index, item)
                        render_next_step_checks(agent, index, item)
                    with remove_col:
                        st.button(
                            ":material/close:",
                            key=f"agent_unstage_{staged['id']}_{index}",
                            type="tertiary",
                            on_click=unstage_item,
                            args=(agent, index),
                            help="Leave this one out",
                        )
                right_col, needs_col = st.columns(2)
                right_col.button(
                    "That's right",
                    key=f"agent_confirm_{staged['id']}",
                    type="primary",
                    icon=":material/check:",
                    on_click=send_staged_to_outbox,
                    args=(search_state, prompts, source_notes),
                    disabled=not staged["items"],
                    use_container_width=True,
                    help=f"Write these to the outbox ({get_outbox_dir()}).",
                )
                needs_col.button(
                    "Needs work",
                    key=f"agent_reject_{staged['id']}",
                    icon=":material/edit:",
                    on_click=start_needs_work,
                    args=(agent,),
                    use_container_width=True,
                    help="Discard this and say what's wrong; then keep refining and stage again.",
                )
        if error := agent.pop("send_error", None):
            st.warning(error)

        if agent.get("needs_work"):
            with st.form("agent_needs_work_form", clear_on_submit=True):
                issue = st.text_area(
                    "What needs work?", placeholder="e.g. Task 2 is Jo's, not mine, and drop the last one."
                )
                submitted = st.form_submit_button("Send", type="primary")
            if submitted and issue.strip():
                rejected = agent.get("rejected") or {"items": []}
                proposal = "\n".join(f"- {describe_staged_item(item)}" for item in rejected["items"]) or "- (nothing)"
                if add_discussion_turn(
                    search_state, f"**Needs work:** {issue.strip()}", model_name, max_tokens, temperature,
                    corpus_prompt, sent_text=f"You proposed:\n{proposal}\n\nThat needs work: {issue.strip()}",
                ):
                    agent["needs_work"], agent["rejected"] = False, None
                    st.rerun()

        waiting = [render_sent_batch(batch) for batch in reversed(agent["sent"])]
        if any(waiting):
            st.button("Check again", icon=":material/refresh:", key="agent_check_sent")


def render_search_results(
    search_state: dict,
    library_folder: Path,
    safe_lib: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    corpus_prompt: str = "",
    model_options: list[str] | None = None,
) -> None:
    """Render the stored results of the last search, with the follow-up discussion alongside."""
    results = search_state["results"]
    if not results:
        st.warning(search_state["notice"] or "No matches found.")
        return

    if search_state["notice"]:
        st.info(search_state["notice"])

    render_library_counts(search_state)

    def hit_location(hit: dict) -> tuple[Path, str, tuple[Path, ...]]:
        """Folder, index name and uploads dirs for a hit (all-library hits carry their own)."""
        return (
            hit.get("_library_folder", library_folder),
            hit.get("_safe_lib", safe_lib),
            hit.get("_uploads_dirs", search_state["uploads_dirs"]),
        )

    # Each source note once, in rank order, with its local path when it can be found.
    source_notes: list[tuple[str, Path | None]] = []
    notes_in_use: list[tuple[str, Path | None]] = []  # minus notes unticked for the discussion
    in_use_keys = {discussion_note_key(r) for r in discussion_results(search_state)}
    seen_sources: set[tuple[str, str]] = set()
    for result in results:
        file_source = str(result.get("file_source", ""))
        source_key = discussion_note_key(result)
        if file_source and source_key not in seen_sources:
            seen_sources.add(source_key)
            result_folder, _result_lib, result_uploads = hit_location(result)
            note = (file_source, resolve_source_path(file_source, result_folder, result_uploads))
            source_notes.append(note)
            if source_key in in_use_keys:
                notes_in_use.append(note)

    main_col, chat_col = st.columns([3, 2], gap="large")

    with main_col:
        on_big_screen = big_screen_turn(search_state)
        if on_big_screen:
            render_big_screen_reply(search_state, *on_big_screen)
            if search_state["answer"]:
                with st.expander("Answer"):
                    st.markdown(link_opportunity_names(search_state["answer"]))
                    render_copy_buttons([("Copy answer", link_opportunity_names(search_state["answer"]))])
        elif search_state["answer"]:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.subheader("Answer", width="content")
                render_copy_buttons([("Copy all output", full_llm_output(search_state))])
            st.markdown(link_opportunity_names(search_state["answer"]))
            render_copy_buttons([
                ("Copy all output", full_llm_output(search_state)),
                ("Copy answer", link_opportunity_names(search_state["answer"])),
                ("Copy question", search_state.get("question", "")),
                ("Copy task prompt", search_state.get("task_prompt", "")),
            ])

            cited_paths = list(dict.fromkeys(path for _file, path in source_notes if path))
            if cited_paths:
                st.caption(
                    "Sources used (expand below to read them): "
                    + ", ".join(f"`{path.name}`" for path in cited_paths)
                )

            usage = search_state["usage"] or {}
            processing_time = usage.get("processing_time")
            if processing_time is not None:
                st.caption(
                    f"Model: `{search_state['model']}` · "
                    f"{usage.get('total', '?')} tokens · "
                    f"{processing_time:.1f}s"
                    + (f" · {usage['tokens_per_second']:.1f} tokens/s" if usage.get("tokens_per_second") else "")
                    + "".join(f" · {part}" for part in search_state.get("prompt_parts", []))
                )
        elif search_state["answer_error"]:
            st.error(f"LLM answer failed: {search_state['answer_error']}")
        elif not search_state["question"]:
            st.info("No question was asked — use the discussion panel to ask about these notes.")

        st.subheader("Sources")
        st.caption("Untick a note to leave it out of the discussion's next replies (and of what an Agent sends).")
        # Group hits by document, keeping the order of each document's best hit.
        hits_by_source: dict[tuple[str, str], list[dict]] = {}
        for i, res in enumerate(results, 1):
            file_source = str(res.get("file_source", "Unknown source"))
            hits_by_source.setdefault((res.get("_library", ""), file_source), []).append({**res, "_rank": i})

        for (library, file_source), hits in hits_by_source.items():
            best = min(
                (h["distance"] for h in hits if h.get("distance") is not None),
                default=None,
            )
            label = f"#{hits[0]['_rank']} — {Path(file_source).name}"
            if hits[0].get("_folder_path"):
                label += f" · {hits[0]['_folder_path']}"
            if library:
                label = f"[{library}] " + label
            if len(hits) > 1:
                label += f" · {len(hits)} passages"
            if best is not None:
                label += f" (distance: {best:.3f})"
            hit_folder, hit_lib, hit_uploads = hit_location(hits[0])
            label += action_badges(hit_lib, file_source)
            note_key = (library, file_source)
            use_key = f"use_note_{search_state.get('id', '')}_{library}_{file_source}"
            if use_key not in st.session_state:
                st.session_state[use_key] = note_key in in_use_keys
            use_col, note_col = st.columns([0.05, 0.95])
            use_col.checkbox(
                "Use in discussion",
                key=use_key,
                on_change=toggle_discussion_note,
                args=(search_state, note_key, use_key),
                label_visibility="collapsed",
                help="Use this note in the discussion and for the Agent. Untick to leave it out of the next replies.",
            )
            with note_col.expander(label if note_key in in_use_keys else f"~~{label}~~ · left out"):
                render_note_actions(hit_lib, file_source)
                render_source_document(
                    file_source,
                    hits,
                    hit_folder,
                    hit_lib,
                    hit_uploads,
                )

    with chat_col:
        follow_up_model, dock = render_follow_up_discussion(
            search_state,
            model_name,
            max_tokens,
            temperature,
            corpus_prompt,
            model_options,
            safe_lib,
        )
        if search_state.get("agent"):
            with dock:  # under the follow-up box, so both stay in view as the answer is scrolled
                render_agent_panel(search_state, notes_in_use, follow_up_model, max_tokens, temperature,
                                   corpus_prompt)


def get_pending_index_settings(safe_lib: str, corpus_config: dict) -> tuple[str, float, bool]:
    """Read embedding settings from Advanced widgets (or saved corpus defaults)."""
    model = st.session_state.get(f"embedding_model_{safe_lib}", corpus_config["embedding_model"])
    threshold = float(
        st.session_state.get(f"distance_threshold_{safe_lib}", corpus_config["distance_threshold"])
    )
    ack = st.session_state.get(
        f"reembed_ack_{safe_lib}",
        st.session_state.get(f"reembed_ack_active_{safe_lib}", True),
    )
    return model, threshold, bool(ack)


def is_first_index_run(active_model: str | None) -> bool:
    """No embedding index yet — initial ingest/rebuild should not require ack."""
    return active_model is None


def render_debug_panel(
    selected_lib: str,
    safe_lib: str,
    status_lib: Library | None,
    corpus_config: dict,
) -> None:
    """Show detailed library diagnostics (caller wraps in sidebar expander)."""
    st.caption(f"Internal library name: `{safe_lib}`")
    st.caption(f"Data directory: `{APP_DATA_DIR}`")
    st.caption(f"SQLite catalog: `{SQLiteConfig.get_uri_string()}`")
    st.caption(f"Vector index path: `{LANCEDB_PATH}`")
    st.caption(f"Corpus folder: `{resolve_library_content_path(get_corpus_root_path(), selected_lib) if get_corpus_root_path() else 'not set'}`")
    st.caption(
        f"Embedding runtime device: **`{resolve_embedding_device_label(get_use_gpu_for_embeddings())}`** · "
        "LLM answers via Ollama use a separate Metal/GPU path."
    )

    corpus_root = get_corpus_root_path()
    if not corpus_root:
        return
    lib_path = resolve_library_content_path(corpus_root, selected_lib)
    try:
        md_files = list(lib_path.rglob("*.md"))
        st.write(f"`.md` files on disk: **{len(md_files)}**")
        if md_files:
            for path in md_files[:5]:
                st.markdown(f"- [`{path.name}`]({path.as_uri()})")
    except Exception as exc:
        st.error(f"Could not scan folder: {exc}")

    if status_lib:
        try:
            card = status_lib.get_library_card()
            if card:
                st.write(
                    f"Ingested: **{card.get('documents', 0)} docs** · "
                    f"**{card.get('blocks', 0)} blocks**"
                )
                md_count = len(list(lib_path.rglob("*.md")))
                if md_count > card.get("documents", 0):
                    st.warning(
                        f"Only **{card.get('documents', 0)}** of **{md_count}** `.md` files are indexed."
                    )
        except Exception:
            st.write("Ingested file count: not available")

        configured_model = corpus_config["embedding_model"]
        active_model = get_active_library_embedding(status_lib)
        try:
            emb_label = format_embedding_status(status_lib, configured_model)
            st.write(f"Embedding index: **{emb_label}**")
            if active_model and active_model != configured_model:
                st.warning(
                    f"Saved config uses `{configured_model}` but index is `{active_model}`."
                )
        except Exception:
            st.write("Embedding index: not detected")


def get_fresh_library_card(safe_lib: str) -> dict | None:
    """Read the latest library card from SQLite (avoids stale in-memory copies)."""
    try:
        card = LibraryCatalog().get_library_card(safe_lib)
        return card if isinstance(card, dict) else None
    except Exception:
        return None


def get_library_block_count(safe_lib: str) -> int:
    """Return the number of text blocks currently stored for a library."""
    card = get_fresh_library_card(safe_lib)
    return int(card.get("blocks", 0)) if card else 0


def get_active_library_embedding(lib: Library) -> str | None:
    card = get_fresh_library_card(lib.library_name)
    if not card:
        return None
    for emb in card.get("embedding", []) or []:
        if emb.get("embedding_status") != "yes" or emb.get("embedding_db") != VECTOR_DB:
            continue
        model = emb.get("embedding_model")
        if model and model not in {"none", ""}:
            return model
    return None


def describe_unbuilt_embedding(blocks: int, vector_rows: int, embedding_model: str) -> str:
    """Explain when ingest finished but vectors were never built."""
    if vector_rows > 0:
        return (
            f"Text index has {blocks} blocks but `{embedding_model}` was never embedded "
            f"({vector_rows} stale vectors from an earlier run remain). "
            "Use **Re-index (full rebuild)** in the sidebar."
        )
    return (
        f"Text index has {blocks} blocks but `{embedding_model}` vectors are not built yet. "
        "Use **Re-index (full rebuild)** in the sidebar."
    )


def get_libraries() -> list[str]:
    """Return selectable libraries for the active corpus mode."""
    corpus_root = get_corpus_root_path()
    if corpus_root is None or not corpus_root.exists():
        return []
    if is_single_corpus_mode():
        return [SINGLE_CORPUS_DISPLAY_NAME] if corpus_has_ingestible_files(corpus_root) else []
    return sorted(d.name for d in corpus_root.iterdir() if d.is_dir())

def sanitize_library_name(name: str) -> str:
    """Convert human-readable folder name to a safe SQLite/file-system token."""
    name = name.replace(" - ", "_").replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^\w]", "", name)
    return name

def load_or_create_library(name: str, force_create: bool = False, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Library:
    """Safely load or create a library using try-except fallback."""
    safe_name = sanitize_library_name(name)

    if force_create:
        return hard_reset_library(name, embedding_model)

    try:
        return Library().load_library(safe_name)
    except Exception:
        return Library().create_new_library(safe_name)


def get_library_folder_path(safe_lib: str) -> Path:
    """Return the on-disk folder for a library under the active account."""
    return APP_DATA_DIR / "accounts" / "llmware" / safe_lib


def count_library_uploads(lib: Library) -> int:
    """Count files already copied into the library uploads folder."""
    try:
        uploads = Path(lib.file_copy_path)
        if not uploads.exists():
            return 0
        return sum(1 for path in uploads.iterdir() if path.is_file())
    except Exception:
        return 0


def clear_library_uploads(lib: Library) -> None:
    """Remove copied source files so ingest will parse them again."""
    try:
        uploads = Path(lib.file_copy_path)
        if uploads.exists():
            shutil.rmtree(uploads)
        uploads.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def repair_library_index_state(lib: Library, embedding_model: str) -> list[str]:
    """Fix split-brain states left by failed ingests or partial resets."""
    actions: list[str] = []
    blocks = get_library_block_count(lib.library_name)
    vectors = get_vector_row_count(lib.library_name, embedding_model)
    uploads = count_library_uploads(lib)

    if blocks == 0 and uploads > 0:
        clear_library_uploads(lib)
        actions.append(f"cleared {uploads} stale upload(s)")

    if blocks == 0 and vectors > 0:
        delete_vector_table(lib.library_name, embedding_model)
        try:
            lib.delete_installed_embedding(embedding_model, VECTOR_DB)
        except Exception:
            pass
        actions.append(f"removed {vectors} orphan vector(s)")

    if blocks > 0 and get_active_library_embedding(lib) is None and vectors > 0:
        delete_vector_table(lib.library_name, embedding_model)
        try:
            lib.delete_installed_embedding(embedding_model, VECTOR_DB)
        except Exception:
            pass
        actions.append(
            f"removed {vectors} stale vector(s) left after an incomplete embedding run"
        )

    return actions


def hard_reset_library(name: str, embedding_model: str) -> Library:
    """Delete library artifacts, vector index, and create a fresh library."""
    safe_name = sanitize_library_name(name)
    library_path = get_library_folder_path(safe_name)
    models_to_clear = {embedding_model}

    try:
        temp_lib = Library().load_library(safe_name)
        active_model = get_active_library_embedding(temp_lib)
        if active_model:
            models_to_clear.add(active_model)
        for model in models_to_clear:
            try:
                temp_lib.delete_installed_embedding(model, VECTOR_DB)
            except Exception:
                pass
            delete_vector_table(safe_name, model)
        temp_lib.delete_library(confirm_delete=True)
    except Exception:
        pass

    if library_path.exists():
        shutil.rmtree(library_path, ignore_errors=True)
    for model in models_to_clear:
        delete_vector_table(safe_name, model)
    return Library().create_new_library(safe_name)


def delete_vector_table(safe_lib: str, embedding_model: str) -> None:
    """Remove the LanceDB table for a library if it exists."""
    try:
        import lancedb

        table_name = get_collection_name(safe_lib, embedding_model)
        lance_dir = LANCEDB_PATH / f"{table_name}.lance"
        if lance_dir.exists():
            shutil.rmtree(lance_dir)

        db = lancedb.connect(str(LANCEDB_PATH))
        if table_name in lancedb_table_names(db):
            db.drop_table(table_name)
    except Exception:
        pass


def should_skip_ingest_path(path: Path) -> bool:
    """Skip macOS metadata folders/files that break PDF/OCR ingest."""
    return any(part == "__MACOSX" or part.startswith("._") for part in path.parts)


def get_ingest_directories(folder_path: Path) -> list[Path]:
    """Return every directory under folder_path that contains ingestible files."""
    directories = {folder_path.resolve()}
    for path in folder_path.rglob("*"):
        if should_skip_ingest_path(path):
            continue
        if path.is_file() and path.suffix.lower() in INGEST_EXTENSIONS:
            directories.add(path.parent.resolve())
    return sorted(
        (d for d in directories if not should_skip_ingest_path(d)),
        key=lambda p: len(p.parts),
    )


def list_ingest_files(folder_path: Path) -> dict[str, tuple[Path, float]]:
    """Ingestible files under folder_path as {file name: (path, modified time)}.

    LLMWare keys a library's files by bare file name, so when two notes in one corpus share a
    name only the first in ingest order (shallowest folder first) is indexed.
    """
    paths = sorted(
        (
            path
            for path in folder_path.rglob("*")
            if path.suffix.lower() in INGEST_EXTENSIONS and not should_skip_ingest_path(path)
        ),
        key=lambda path: (len(path.parts), str(path)),
    )
    files: dict[str, tuple[Path, float]] = {}
    for path in paths:
        if path.name in files:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.is_file():
            files[path.name] = (path, stat.st_mtime)
    return files


def plan_text_rescan(lib: Library, folder_path: Path) -> dict[str, list]:
    """Compare folder_path with the library: notes that are new, changed or removed.

    LLMWare copies each file into the library's uploads folder when it indexes it, so the copy's
    modified time is when the note was indexed. A note whose file is newer has been edited since.
    Costs two stats per note; no file is read.
    """
    indexed: dict[str, float] = {}
    uploads = Path(lib.file_copy_path)
    if uploads.is_dir():
        for copy in uploads.iterdir():
            if copy.suffix.lower() in INGEST_EXTENSIONS and copy.is_file():
                indexed[copy.name] = copy.stat().st_mtime

    on_disk = list_ingest_files(folder_path)
    new = [path for name, (path, _mtime) in on_disk.items() if name not in indexed]
    changed = [path for name, (path, mtime) in on_disk.items() if name in indexed and mtime > indexed[name]]
    # An empty or missing folder (unmounted drive, renamed root) must not wipe the index.
    removed = [name for name in indexed if name not in on_disk] if on_disk else []
    return {"new": new, "changed": changed, "removed": removed}


def delete_block_vectors(safe_lib: str, rowids: list[int]) -> None:
    """Drop the vectors of deleted blocks — SQLite can hand their row ids to new blocks."""
    if not rowids:
        return
    card = get_fresh_library_card(safe_lib) or {}
    models = {
        emb.get("embedding_model")
        for emb in card.get("embedding") or []
        if emb.get("embedding_db") == VECTOR_DB and emb.get("embedding_model") not in {None, "", "none"}
    }
    for model in models:
        name = get_collection_name(safe_lib, model)
        if not (LANCEDB_PATH / f"{name}.lance").is_dir():
            continue
        table = get_lancedb_connection().open_table(name)
        for start in range(0, len(rowids), 500):
            ids = ", ".join(f"'{rowid}'" for rowid in rowids[start:start + 500])
            table.delete(f"id IN ({ids})")


def remove_indexed_notes(lib: Library, file_names: list[str]) -> None:
    """Delete notes' blocks, vectors and uploads copies, so they can be re-ingested or dropped."""
    if not file_names:
        return
    safe_lib = lib.library_name
    table = safe_lib.replace('"', '""')
    rowids: list[int] = []
    with sqlite3.connect(SQLiteConfig.get_uri_string()) as conn:
        for start in range(0, len(file_names), 500):
            chunk = file_names[start:start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            rowids += [
                row[0]
                for row in conn.execute(f'SELECT rowid FROM "{table}" WHERE file_source IN ({placeholders})', chunk)
            ]
            conn.execute(f'DELETE FROM "{table}" WHERE file_source IN ({placeholders})', chunk)
        # add_files reports what it added from changes in these card counts.
        blocks, documents = conn.execute(f'SELECT COUNT(*), COUNT(DISTINCT doc_ID) FROM "{table}"').fetchone()
        conn.execute(
            "UPDATE library SET blocks = ?, documents = ? WHERE library_name = ?", (blocks, documents, safe_lib)
        )
    delete_block_vectors(safe_lib, rowids)
    uploads = Path(lib.file_copy_path)
    for name in file_names:
        (uploads / name).unlink(missing_ok=True)


def ingest_all_files(lib: Library, folder_path: Path) -> dict:
    """Bring the text index in line with folder_path: add new notes, re-ingest edited ones and
    drop removed ones. Embeddings are not touched; new blocks wait for the next embedding run.

    LLMWare's add_files skips any file name it has seen, so on its own it never picks up edits,
    and it only scans one directory level at a time.
    """
    totals = {
        "docs_added": 0,
        "blocks_added": 0,
        "dirs_scanned": 0,
        "rejected_files": [],
        "skipped_dirs": 0,
    }
    # Full-corpus scans are mostly markdown; skip standalone image OCR to reduce open files.
    add_files_kwargs = {}
    if load_app_settings().get("corpus_mode") == CORPUS_MODE_SINGLE:
        add_files_kwargs = {"get_images": False, "verbose_level": 0}

    plan = plan_text_rescan(lib, folder_path)
    totals.update(
        notes_new=len(plan["new"]), notes_changed=len(plan["changed"]), notes_removed=len(plan["removed"])
    )
    remove_indexed_notes(lib, [path.name for path in plan["changed"]] + plan["removed"])

    directories = sorted(
        {path.parent for path in plan["new"] + plan["changed"]}, key=lambda d: (len(d.parts), str(d))
    )
    for index, directory in enumerate(directories):
        try:
            result = lib.add_files(str(directory), **add_files_kwargs)
        except Exception as exc:
            totals["skipped_dirs"] += 1
            totals.setdefault("ingest_warnings", []).append(
                f"`{directory.name}`: {format_exception_for_user(exc)}"
            )
            gc.collect()
            continue
        totals["dirs_scanned"] += 1
        if not result:
            gc.collect()
            continue
        totals["docs_added"] += result.get("docs_added", 0) or 0
        totals["blocks_added"] += result.get("blocks_added", 0) or 0
        rejected = result.get("rejected_files") or []
        if isinstance(rejected, list):
            totals["rejected_files"].extend(rejected)
        if index % 25 == 24:
            gc.collect()

    return totals


def prepare_embeddings(lib: Library, embedding_model: str) -> None:
    """Align embedding metadata with the current text index before building vectors."""
    blocks = get_library_block_count(lib.library_name)
    vectors = get_vector_row_count(lib.library_name, embedding_model)

    if blocks == 0:
        delete_vector_table(lib.library_name, embedding_model)
        try:
            lib.delete_installed_embedding(embedding_model, VECTOR_DB)
        except Exception:
            pass
        return

    ready, _ = is_embedding_ready(lib, embedding_model)
    if ready:
        return

    if vectors > 0:
        delete_vector_table(lib.library_name, embedding_model)
    try:
        lib.delete_installed_embedding(embedding_model, VECTOR_DB)
    except Exception:
        delete_vector_table(lib.library_name, embedding_model)


def sync_library(lib: Library, folder_path: Path, embedding_model: str) -> dict:
    """Ingest all files recursively, then build embeddings."""
    repair_library_index_state(lib, embedding_model)
    ingest_stats = ingest_all_files(lib, folder_path)
    record_text_rescan({lib.library_name: ingest_stats}, "manual")
    prepare_embeddings(lib, embedding_model)
    lib.install_new_embedding(
        embedding_model,
        VECTOR_DB,
        batch_size=get_embedding_batch_size(embedding_model),
        use_gpu=_EMBEDDING_USE_GPU,
    )
    ingest_stats["vectors"] = get_vector_row_count(lib.library_name, embedding_model)
    ingest_stats["blocks_total"] = get_library_block_count(lib.library_name)
    if ingest_stats["vectors"] == 0 and ingest_stats.get("blocks_total", 0) > 0:
        raise RuntimeError(
            f"Embedding finished with 0 vectors for `{embedding_model}` despite "
            f"{ingest_stats['blocks_total']} text blocks. Try **Re-index** again or "
            "switch to a smaller model (e.g. mini-lm-sbert) to confirm the pipeline."
        )
    return ingest_stats


def load_text_rescan_status() -> dict:
    try:
        return json.loads(TEXT_RESCAN_STATUS_PATH.read_text())
    except Exception:
        return {}


def record_text_rescan(results: dict[str, dict], trigger: str, errors: list[str] | None = None) -> None:
    """Save when each library's text index was last brought up to date. Call with the index lock held."""
    status = load_text_rescan_status()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    libraries = status.setdefault("libraries", {})
    totals = {"new": 0, "changed": 0, "removed": 0}
    for safe_lib, stats in results.items():
        counts = {key: int(stats.get(f"notes_{key}", 0)) for key in totals}
        libraries[safe_lib] = {"at": now, "trigger": trigger, **counts}
        for key, value in counts.items():
            totals[key] += value
    status["last_run"] = {"at": now, "trigger": trigger, "libraries": len(results), **totals, "errors": errors or []}
    tmp_path = TEXT_RESCAN_STATUS_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(status, indent=2))
    tmp_path.replace(TEXT_RESCAN_STATUS_PATH)


def resolve_text_rescan_targets(all_corpora: bool) -> list[tuple[str, str, Path]]:
    """(display name, library key, folder) to rescan, from saved settings — safe off the script thread."""
    settings = load_app_settings()
    corpus_root = load_corpus_root_path()
    if corpus_root is None or not corpus_root.is_dir():
        return []
    if settings.get("corpus_mode") == CORPUS_MODE_SINGLE:
        return [(SINGLE_CORPUS_DISPLAY_NAME, SINGLE_CORPUS_LIBRARY_NAME, corpus_root)]
    names = sorted(d.name for d in corpus_root.iterdir() if d.is_dir())
    if all_corpora:
        names = filter_libraries_for_display(names)
    else:
        names = [name for name in names if name == settings.get("last_selected_library")]
    return [(name, sanitize_library_name(name), corpus_root / name) for name in names]


def run_text_rescan(targets: list[tuple[str, str, Path]], trigger: str) -> dict:
    """Text-only rescan (no embeddings) of each target. Call with the index lock held."""
    results: dict[str, dict] = {}
    errors: list[str] = []
    for display_name, safe_name, folder in targets:
        try:
            lib = load_or_create_library(safe_name)
            results[safe_name] = ingest_all_files(lib, folder)
        except Exception as exc:
            errors.append(f"{display_name}: {format_exception_for_user(exc)}")
        gc.collect()
    record_text_rescan(results, trigger, errors)
    return {"results": results, "errors": errors}


class TextRescanScheduler:
    """Background text rescan every AUTO_RESCAN_SECONDS while the Streamlit server runs.

    `lock` is held by every index write (auto rescans, Rescan, Re-index, rebuilds), so an auto
    rescan skips its turn while a manual scan runs and a manual scan waits for an auto rescan.
    """

    def __init__(self):
        # A cleared resource cache builds a new scheduler; retire the old thread but share its lock.
        previous = [
            thread.rescan_scheduler for thread in threading.enumerate() if hasattr(thread, "rescan_scheduler")
        ]
        for old in previous:
            old.stop.set()
            old.wake.set()
        self.lock = previous[0].lock if previous else threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="text-rescan", daemon=True)
        self.thread.rescan_scheduler = self
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.is_set():
            self.wake.wait(AUTO_RESCAN_SECONDS)
            self.wake.clear()
            settings = load_app_settings()
            if self.stop.is_set() or not settings.get("auto_rescan_enabled"):
                continue
            if not self.lock.acquire(blocking=False):
                continue
            try:
                run_text_rescan(resolve_text_rescan_targets(bool(settings.get("auto_rescan_all_corpora"))), "auto")
            except Exception as exc:
                record_text_rescan({}, "auto", [format_exception_for_user(exc)])
            finally:
                self.lock.release()


@st.cache_resource
def get_text_rescan_scheduler() -> TextRescanScheduler:
    return TextRescanScheduler()


def persist_auto_rescan_settings() -> None:
    enabled = bool(st.session_state.get("auto_rescan_enabled", False))
    save_app_settings(
        {
            "auto_rescan_enabled": enabled,
            "auto_rescan_all_corpora": bool(st.session_state.get("auto_rescan_all_corpora", False)),
        }
    )
    if enabled:
        get_text_rescan_scheduler().wake.set()


def format_rescan_time(iso_time: str) -> str:
    when = datetime.datetime.fromisoformat(iso_time)
    if when.date() == datetime.date.today():
        return when.strftime("%H:%M")
    return when.strftime("%b %d %H:%M")


def describe_last_text_rescan(safe_libs: list[str]) -> str:
    """One line for the search box: when these libraries' text was last rescanned, and auto-rescan state."""
    status = load_text_rescan_status()
    entries = [status.get("libraries", {}).get(key) for key in safe_libs]
    if not entries or not all(entries):
        text = "Not rescanned since change tracking was added"
    elif len(entries) == 1:
        entry = entries[0]
        text = (
            f"Last rescan {format_rescan_time(entry['at'])} ({entry.get('trigger', 'manual')}) · "
            f"{entry.get('new', 0)} new, {entry.get('changed', 0)} changed, {entry.get('removed', 0)} removed"
        )
    else:
        oldest = min(entry["at"] for entry in entries)
        text = f"All {len(entries)} libraries rescanned since {format_rescan_time(oldest)}"

    settings = load_app_settings()
    if settings.get("auto_rescan_enabled"):
        scope = "all visible corpora" if settings.get("auto_rescan_all_corpora") else "this corpus"
        text += f" · auto every {AUTO_RESCAN_SECONDS // 60} min ({scope})"
    else:
        text += " · auto-rescan off"
    errors = status.get("last_run", {}).get("errors") or []
    if errors:
        text += f" · ⚠️ last run: {errors[0]}"
    return text


def get_collection_name(safe_lib: str, embedding_model: str) -> str:
    """Return the LanceDB table name LLMWare uses for a library."""
    utils = _EmbeddingUtils(
        library_name=safe_lib,
        model_name=embedding_model,
        account_name="llmware",
        db_name=VECTOR_DB,
        embedding_dims=get_model_dims(embedding_model),
    )
    return utils.create_safe_collection_name()


def get_vector_row_count(safe_lib: str, embedding_model: str) -> int:
    """Return the number of vectors stored for a library, or 0 if missing."""
    try:
        name = get_collection_name(safe_lib, embedding_model)
        if not (LANCEDB_PATH / f"{name}.lance").is_dir():
            return 0
        db = get_lancedb_connection()
        return db.open_table(name).count_rows()
    except Exception:
        return 0


def migrate_legacy_vectors(safe_lib: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> bool:
    """Copy vectors from /tmp/lancedb when the persistent index is empty."""
    if not LEGACY_LANCEDB_PATH.exists():
        return False

    if get_vector_row_count(safe_lib, embedding_model) > 0:
        return False

    name = get_collection_name(safe_lib, embedding_model)
    src = LEGACY_LANCEDB_PATH / f"{name}.lance"
    if not src.exists():
        return False

    try:
        import lancedb

        legacy_db = lancedb.connect(str(LEGACY_LANCEDB_PATH))
        if name not in lancedb_table_names(legacy_db) or legacy_db.open_table(name).count_rows() == 0:
            return False

        dest = LANCEDB_PATH / f"{name}.lance"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return True
    except Exception:
        return False


def migrate_all_legacy_vectors() -> list[str]:
    """One-time migration of any legacy /tmp indexes into the persistent store."""
    if not LEGACY_LANCEDB_PATH.exists():
        return []

    migrated: list[str] = []
    try:
        import lancedb

        legacy_db = lancedb.connect(str(LEGACY_LANCEDB_PATH))
        dest_db = lancedb.connect(str(LANCEDB_PATH))

        for name in lancedb_table_names(legacy_db):
            if legacy_db.open_table(name).count_rows() == 0:
                continue
            if name in lancedb_table_names(dest_db) and dest_db.open_table(name).count_rows() > 0:
                continue

            src = LEGACY_LANCEDB_PATH / f"{name}.lance"
            if not src.exists():
                continue

            dest = LANCEDB_PATH / f"{name}.lance"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            migrated.append(name)
    except Exception:
        return migrated

    return migrated


def is_embedding_ready(lib: Library, embedding_model: str) -> tuple[bool, str]:
    """Return whether the library has a usable semantic index for our model/db."""
    safe_lib = lib.library_name
    card = get_fresh_library_card(safe_lib)
    blocks = int(card.get("blocks", 0)) if card else 0
    vector_rows = get_vector_row_count(safe_lib, embedding_model)

    if blocks == 0:
        if vector_rows > 0:
            return False, (
                f"Text index is empty but {vector_rows} stale vectors remain. "
                "Use **Re-index (full rebuild)** in the sidebar."
            )
        if count_library_uploads(lib) > 0:
            return False, (
                "Files were copied but never parsed into the text index. "
                "Use **Re-index (full rebuild)** in the sidebar."
            )
        return False, "No text blocks indexed yet. Use **Re-index (full rebuild)** in the sidebar."

    record = card.get("embedding", []) if card else []
    if not record:
        return False, describe_unbuilt_embedding(blocks, vector_rows, embedding_model)

    for emb in record:
        model = emb.get("embedding_model")
        if model in {None, "", "none"}:
            continue
        if model != embedding_model:
            continue
        if emb.get("embedding_db") != VECTOR_DB:
            continue
        if emb.get("embedding_status") != "yes":
            return False, (
                f"Embedding for `{embedding_model}` did not complete "
                f"(status: {emb.get('embedding_status')}). "
                "Use **Re-index (full rebuild)** in the sidebar."
            )

        expected = emb.get("embedded_blocks", 0)
        vector_rows = get_vector_row_count(safe_lib, embedding_model)
        if vector_rows == 0:
            return False, describe_unbuilt_embedding(blocks, vector_rows, embedding_model)
        if vector_rows < expected:
            return True, f"{vector_rows}/{expected} vectors indexed ({embedding_model})"
        return True, f"{vector_rows} blocks indexed ({embedding_model})"

    active = get_active_library_embedding(lib)
    if active and active != embedding_model:
        return False, f"Index uses `{active}`. Rebuild to switch to `{embedding_model}`."

    return False, describe_unbuilt_embedding(blocks, vector_rows, embedding_model)


def format_embedding_status(lib: Library, embedding_model: str) -> str:
    ready, message = is_embedding_ready(lib, embedding_model)
    return f"Ready — {message}" if ready else f"Not ready — {message}"


def get_embedding_progress(safe_lib: str, embedding_model: str) -> dict | None:
    """Read live embedding progress from LLMWare's status table."""
    status_list = Status().get_embedding_status(safe_lib, embedding_model)
    if not status_list:
        return None
    return status_list[0]


def run_with_embedding_progress(
    operation_label: str, safe_lib: str, embedding_model: str, work_fn, container=None
) -> dict:
    """Run sync/rebuild work in a background thread and poll embedding progress every 5s."""
    container = container or st.sidebar
    error: list[Exception | None] = [None]
    ingest_stats: list[dict] = [{}]
    index_lock = get_text_rescan_scheduler().lock

    def worker() -> None:
        try:
            with index_lock:
                ingest_stats[0] = work_fn() or {}
        except Exception as exc:
            error[0] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    progress_bar = container.progress(0.0, text=f"{operation_label}: starting…")
    live_status = container.empty()

    while True:
        entry = get_embedding_progress(safe_lib, embedding_model)
        if entry and int(entry.get("total", 0)) > 0:
            current = int(entry.get("current", 0))
            total = max(int(entry.get("total", 0)), 1)
            summary = entry.get("summary", f"{current} of {total} blocks")
            vector_rows = get_vector_row_count(safe_lib, embedding_model)
            pct = min(current / total, 1.0)
            progress_text = f"{operation_label}: {summary} · {vector_rows} vectors stored"
            progress_bar.progress(pct, text=progress_text)
            live_status.caption(progress_text)
        else:
            blocks = get_library_block_count(safe_lib)
            progress_text = f"{operation_label}: ingesting files… ({blocks} blocks indexed so far)"
            progress_bar.progress(0.0, text=progress_text)
            live_status.caption(progress_text)

        if not thread.is_alive():
            break
        time.sleep(EMBEDDING_POLL_SECONDS)

    thread.join()

    if error[0]:
        raise error[0]

    entry = get_embedding_progress(safe_lib, embedding_model)
    vectors = get_vector_row_count(safe_lib, embedding_model)
    blocks_total = get_library_block_count(safe_lib)
    embed_current = blocks_total
    embed_total = blocks_total
    if entry and int(entry.get("total", 0)) > 0:
        embed_current = int(entry.get("current", 0))
        embed_total = max(int(entry.get("total", 0)), 1)
        summary = entry.get("summary", "complete")
        progress_text = f"{operation_label}: {summary} · {vectors} vectors stored"
    else:
        summary = "complete"
        progress_text = f"{operation_label}: complete · {vectors} vectors stored"

    progress_bar.progress(1.0, text=progress_text)
    live_status.success(progress_text)

    stats = ingest_stats[0]
    return {
        "operation": operation_label,
        "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": embedding_model,
        "message": progress_text,
        "progress_text": progress_text,
        "progress_pct": 1.0,
        "summary": summary,
        "embedding_current": embed_current,
        "embedding_total": embed_total,
        "docs_added": stats.get("docs_added", 0),
        "blocks_added": stats.get("blocks_added", 0),
        "dirs_scanned": stats.get("dirs_scanned", 0),
        "blocks_total": stats.get("blocks_total", blocks_total),
        "vectors": stats.get("vectors", vectors),
    }


def run_multi_corpus_rebuild(
    libraries: list[str], corpus_root: Path, embedding_model: str, container=None, text_only: bool = False
) -> dict:
    """Full hard rebuild of every sub-corpus, one after another, on a single embedding model.

    With `text_only`, only the text index is rebuilt; embeddings wait for each corpus's next Rescan.
    """
    container = container or st.sidebar
    total = len(libraries)
    overall_bar = container.progress(0.0, text=f"Multi-corpus rebuild: 0/{total} complete")
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for index, lib_name in enumerate(libraries, start=1):
        safe_name = sanitize_library_name(lib_name)
        content_path = corpus_root / lib_name
        overall_bar.progress(
            (index - 1) / total,
            text=f"Multi-corpus rebuild: {index - 1}/{total} complete · now '{lib_name}'",
        )

        def _sync_rebuild(name: str = safe_name, path: Path = content_path) -> dict:
            lib = load_or_create_library(name, force_create=True, embedding_model=embedding_model)
            if text_only:
                stats = ingest_all_files(lib, path)
                record_text_rescan({name: stats}, "rebuild")
                return stats
            return sync_library(lib, path, embedding_model)

        try:
            job_status = run_with_embedding_progress(
                f"Rebuilding '{lib_name}' ({index}/{total})",
                safe_name,
                embedding_model,
                _sync_rebuild,
                container,
            )
            commit_job_status(job_status, lib_name, safe_name)
            save_corpus_config(safe_name, {"embedding_model": embedding_model, "vector_db": VECTOR_DB})
            succeeded.append(lib_name)
        except Exception as exc:
            failed.append((lib_name, format_exception_for_user(exc)))

        # Drop stale per-corpus widget values so the new saved model is picked up.
        st.session_state.pop(f"embedding_model_{safe_name}", None)

    overall_bar.progress(1.0, text=f"Multi-corpus rebuild: {len(succeeded)}/{total} succeeded")
    return {
        "embedding_model": embedding_model,
        "text_only": text_only,
        "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "succeeded": succeeded,
        "failed": failed,
    }


def run_multi_corpus_rescan(libraries: list[str], corpus_root: Path, container=None) -> dict:
    """Incremental rescan of every sub-corpus, one after another, each on its own embedding engine.

    New and edited notes are (re-)ingested and embedded, removed ones dropped; nothing is wiped.
    """
    container = container or st.sidebar
    total = len(libraries)
    overall_bar = container.progress(0.0, text=f"Multi-corpus rescan: 0/{total} complete")
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for index, lib_name in enumerate(libraries, start=1):
        safe_name = sanitize_library_name(lib_name)
        content_path = corpus_root / lib_name
        overall_bar.progress(
            (index - 1) / total,
            text=f"Multi-corpus rescan: {index - 1}/{total} complete · now '{lib_name}'",
        )
        try:
            lib = load_or_create_library(safe_name)
            # Keep the engine the corpus is embedded with, so no corpus needs a full re-embed.
            embedding_model = get_active_library_embedding(lib) or get_corpus_config(safe_name)["embedding_model"]
            ensure_embedding_dependencies(embedding_model)

            def _sync(lib: Library = lib, path: Path = content_path, model: str = embedding_model) -> dict:
                return sync_library(lib, path, model)

            job_status = run_with_embedding_progress(
                f"Rescanning '{lib_name}' ({index}/{total})", safe_name, embedding_model, _sync, container
            )
            commit_job_status(job_status, lib_name, safe_name)
            succeeded.append(lib_name)
        except Exception as exc:
            failed.append((lib_name, format_exception_for_user(exc)))

    overall_bar.progress(1.0, text=f"Multi-corpus rescan: {len(succeeded)}/{total} succeeded")
    return {
        "operation": "rescan",
        "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "succeeded": succeeded,
        "failed": failed,
    }


def render_multi_corpus_rebuild_summary() -> None:
    summary = st.session_state.get("multi_rebuild_summary")
    if not summary:
        return
    succeeded, failed = summary["succeeded"], summary["failed"]
    total = len(succeeded) + len(failed)
    if summary.get("operation") == "rescan":
        message = (
            f"Multi-corpus rescan: {len(succeeded)}/{total} corpora rescanned · {summary['completed_at']}"
        )
    else:
        engine = "text only" if summary.get("text_only") else f"`{summary['embedding_model']}`"
        message = (
            f"Multi-corpus rebuild on {engine}: "
            f"{len(succeeded)}/{total} corpora rebuilt · {summary['completed_at']}"
        )
    if summary.get("text_only"):
        message += " — use **⚡ Rescan** in a corpus to rebuild its embeddings"
    if failed:
        st.sidebar.warning(message)
        for lib_name, reason in failed:
            st.sidebar.caption(f"❌ **{lib_name}** — {reason}")
    else:
        st.sidebar.success(message)


# ---------------------------------------------------------------------------
# GETTING STARTED WIZARD (first run, or after reset_rag.sh)
# ---------------------------------------------------------------------------

def has_indexed_library() -> bool:
    """True once any library has text blocks — i.e. the app has been set up before."""
    try:
        cards = LibraryCatalog().all_library_cards() or []
    except Exception:
        return False
    return any(int(card.get("blocks", 0) or 0) > 0 for card in cards if isinstance(card, dict))


def should_show_setup_wizard() -> bool:
    if st.session_state.get("setup_wizard_done"):
        return False
    return not has_indexed_library()


@st.cache_data(ttl=60, max_entries=8, show_spinner="Counting files…")
def scan_corpus_folders(corpus_root: str) -> tuple[int, dict[str, int]]:
    """Count ingestible files in the whole tree and in each top-level folder."""
    root = Path(corpus_root)
    per_folder = {d.name: 0 for d in root.iterdir() if d.is_dir()}
    total = 0
    for path in root.rglob("*"):
        if should_skip_ingest_path(path) or not path.is_file():
            continue
        if path.suffix.lower() not in INGEST_EXTENSIONS:
            continue
        total += 1
        parts = path.relative_to(root).parts
        if len(parts) > 1 and parts[0] in per_folder:
            per_folder[parts[0]] += 1
    return total, dict(sorted(per_folder.items()))


def wizard_browse_for_root() -> None:
    picked = browse_for_directory(st.session_state.get("wizard_corpus_root") or None)
    if picked:
        st.session_state["wizard_corpus_root"] = picked.strip()


def dismiss_setup_wizard() -> None:
    st.session_state["setup_wizard_done"] = True


def render_setup_wizard() -> None:
    """Three-step getting started guide: notes folder, indexing approach, embedding engine."""
    if "wizard_corpus_root" not in st.session_state:
        saved_root = load_corpus_root_path()
        st.session_state["wizard_corpus_root"] = str(saved_root) if saved_root else ""

    st.header("Getting started")
    st.markdown(
        "Welcome! Three quick steps and your notes will be searchable by meaning, not just keywords. "
        "This guide appears on first run and after a reset; everything here can be changed later in the sidebar."
    )

    with st.container(border=True):
        st.subheader("1. Choose your notes folder")
        st.markdown(
            "Point the app at a folder of notes. Any of these work:\n"
            f"- **[Local Apple Notes Export]({LOCAL_NOTES_EXPORT_URL})** — our companion app. "
            "Exports your Apple Notes to Markdown folders on this Mac.\n"
            f"- **[Falcon Notes Exporter]({FALCON_NOTES_EXPORTER_URL})** — another way to export Apple Notes to Markdown.\n"
            "- **Any other Markdown corpus** — an Obsidian vault, a docs folder, a repo of notes. "
            "PDF, Word, PowerPoint, Excel, CSV, JSON and plain-text files are indexed too."
        )
        st.caption(
            "Using an exporter? Pick the folder that holds one subfolder per account or category "
            "(often called `iCloud`)."
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.text_input(
                "Notes folder",
                key="wizard_corpus_root",
                placeholder="/Users/you/Documents/notes-export/iCloud",
            )
            st.button("Browse…", icon=":material/folder_open:", on_click=wizard_browse_for_root)

        raw_root = str(st.session_state.get("wizard_corpus_root", "")).strip()
        if not raw_root:
            st.info("Choose a folder to continue.")
            return
        corpus_root = Path(raw_root).expanduser()
        if not corpus_root.is_dir():
            st.error(f"Folder not found: `{corpus_root}`")
            return
        total_files, folder_counts = scan_corpus_folders(str(corpus_root))
        if not total_files:
            st.warning(
                "No files the app can index were found here. "
                f"Supported types: {', '.join(sorted(INGEST_EXTENSIONS))}"
            )
            return
        filled_folders = {name: count for name, count in folder_counts.items() if count}
        st.success(
            f"Found **{total_files:,}** files to index"
            + (f" across **{len(filled_folders)}** top-level folders." if filled_folders else ".")
        )

    with st.container(border=True):
        st.subheader("2. Choose how to index")
        choices = {
            SETUP_TRIAL: (
                "Try a small corpus first — fastest",
                "Index one top-level folder so you can see how search feels in a few minutes. "
                "Add the other folders whenever you're ready.",
            ),
            SETUP_WHOLE: (
                "Index everything as one corpus — simplest, but slowest",
                "One search across all your notes. The first build takes longest, and changing the "
                "embedding engine means re-indexing the lot.",
            ),
            SETUP_PER_FOLDER: (
                "Index each top-level folder as its own corpus — most flexible",
                "Each folder gets its own database. Search them one at a time, and re-index, hide or "
                "change the engine for one folder without touching the others.",
            ),
        }
        options = [SETUP_TRIAL, SETUP_WHOLE, SETUP_PER_FOLDER] if filled_folders else [SETUP_WHOLE]
        choice = st.radio(
            "Indexing approach",
            options,
            format_func=lambda key: choices[key][0],
            captions=[choices[key][1] for key in options],
            key="wizard_choice",
            label_visibility="collapsed",
        )
        trial_folder = None
        if not filled_folders:
            st.caption("Your notes aren't split into subfolders, so they'll be indexed as one corpus.")
        elif choice == SETUP_TRIAL:
            folder_names = list(filled_folders)
            smallest = min(folder_names, key=lambda name: filled_folders[name])
            trial_folder = st.selectbox(
                "Folder to try",
                folder_names,
                index=folder_names.index(smallest),
                format_func=lambda name: f"{name} — {filled_folders[name]:,} files",
                key="wizard_trial_folder",
            )
            st.caption(
                "Later, index the rest from the sidebar one at a time, or all at once with "
                "**Rebuild all corpora** in Advanced."
            )
        elif choice == SETUP_PER_FOLDER:
            st.caption("Folders are indexed one after another: " + " → ".join(filled_folders))
            empty_folders = [name for name, count in folder_counts.items() if not count]
            if empty_folders:
                st.caption(f"Skipping {len(empty_folders)} empty folder(s): {', '.join(empty_folders)}")
        else:
            st.caption(f"All {total_files:,} files will go into a single corpus called **{SINGLE_CORPUS_DISPLAY_NAME}**.")

    with st.container(border=True):
        st.subheader("3. Choose an embedding engine")
        catalog = load_embedding_model_catalog()
        labels = {model["id"]: model["label"] for model in catalog}
        model_options = [model_id for model_id in RECOMMENDED_EMBEDDING_MODELS if model_id in labels]
        embedding_model = st.selectbox(
            "Embedding engine",
            model_options,
            index=model_options.index(DEFAULT_EMBEDDING_MODEL) if DEFAULT_EMBEDDING_MODEL in model_options else 0,
            format_func=lambda model_id: format_embedding_model_option(model_id, labels),
            key="wizard_embedding_model",
        )
        st.caption(
            f"Not sure? Keep **{DEFAULT_EMBEDDING_MODEL}** — it's the quickest. Larger engines find subtler "
            "matches but take longer to build. Models marked **(g)** can use this machine's GPU."
        )
        missing_deps = get_missing_embedding_dependencies(embedding_model)
        if missing_deps:
            render_embedding_dependency_install_help(missing_deps, embedding_model)
        use_gpu = st.checkbox(
            "Use GPU for embeddings",
            value=bool(load_app_settings().get("use_gpu_for_embeddings", False)),
            key="wizard_use_gpu",
            help="Faster on models marked (g). Falls back to CPU automatically if the GPU can't run a model.",
        )

    with st.container(horizontal=True, vertical_alignment="center"):
        start = st.button(
            "Start indexing",
            type="primary",
            icon=":material/play_arrow:",
            disabled=bool(missing_deps),
            key="wizard_start",
        )
        st.button("Skip for now", type="tertiary", on_click=dismiss_setup_wizard, key="wizard_skip")

    if start:
        run_setup_wizard(corpus_root, choice, trial_folder, list(filled_folders), embedding_model, use_gpu)


def run_setup_wizard(
    corpus_root: Path,
    choice: str,
    trial_folder: str | None,
    folders: list[str],
    embedding_model: str,
    use_gpu: bool,
) -> None:
    """Save the wizard's choices, then index straight away with progress shown on the page."""
    mode = CORPUS_MODE_SINGLE if choice == SETUP_WHOLE else CORPUS_MODE_MULTI
    save_app_settings(
        {"corpus_root_path": str(corpus_root), "corpus_mode": mode, "use_gpu_for_embeddings": use_gpu}
    )
    st.session_state["corpus_root_path_input"] = str(corpus_root)
    st.session_state["corpus_mode_toggle"] = mode

    progress = st.container()
    try:
        ensure_embedding_dependencies(embedding_model)
        set_embedding_acceleration(use_gpu)

        if choice == SETUP_PER_FOLDER:
            summary = run_multi_corpus_rebuild(folders, corpus_root, embedding_model, progress)
            st.session_state["multi_rebuild_summary"] = summary
            if not summary["succeeded"]:
                progress.error("None of the folders could be indexed:")
                for lib_name, reason in summary["failed"]:
                    progress.caption(f"❌ **{lib_name}** — {reason}")
                return
            first_library = summary["succeeded"][0]
        else:
            display_name = trial_folder or SINGLE_CORPUS_DISPLAY_NAME
            safe_name = sanitize_library_name(trial_folder) if trial_folder else SINGLE_CORPUS_LIBRARY_NAME
            content_path = corpus_root / trial_folder if trial_folder else corpus_root

            def _index() -> dict:
                lib = load_or_create_library(safe_name, force_create=True, embedding_model=embedding_model)
                return sync_library(lib, content_path, embedding_model)

            job_status = run_with_embedding_progress(
                f"Indexing '{display_name}'", safe_name, embedding_model, _index, progress
            )
            commit_job_status(job_status, display_name, safe_name)
            save_corpus_config(safe_name, {"embedding_model": embedding_model, "vector_db": VECTOR_DB})
            first_library = trial_folder
    except Exception as exc:
        handle_embedding_run_error(exc, embedding_model)
        return

    if first_library:
        save_app_settings({"last_selected_library": first_library})
        st.session_state["selected_library"] = first_library
    st.session_state["setup_wizard_done"] = True
    st.rerun()


def get_last_updated(corpus_folder: Path) -> str | None:
    """Return human-readable timestamp of local DB/Lance files."""
    candidates = list(corpus_folder.rglob("*.db")) + list(corpus_folder.rglob("*.lance"))
    if not candidates:
        return None
    latest_mtime = max(f.stat().st_mtime for f in candidates)
    return datetime.datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M")


def discover_ollama_models(host: str = OLLAMA_HOST, port: int = OLLAMA_PORT) -> tuple[bool, list[str]]:
    """Return whether Ollama is reachable and which models are installed locally."""
    try:
        import requests

        response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        response.raise_for_status()
        models = [entry["name"] for entry in response.json().get("models", [])]
        return True, sorted(models)
    except Exception:
        return False, []


def build_context_from_results(results: list, max_chars: int = LLM_CONTEXT_MAX_CHARS) -> str:
    """Pack retrieved note chunks into a prompt context block for the LLM."""
    parts: list[str] = []
    total = 0

    for index, result in enumerate(results, 1):
        source = Path(str(result.get("file_source", "Unknown"))).name
        text = str(result.get("text", "")).strip()
        if not text:
            continue

        folder = result.get("_folder_path")  # where the note lives, so the LLM knows what it's about
        chunk = f"[Source {index}: {source}" + (f", in {folder}" if folder else "") + f"]\n{text}\n"
        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts)


def build_llm_prompt(
    question: str,
    results: list,
    max_tokens: int = 512,
    temperature: float = 0.3,
    corpus_prompt: str = "",
    task_prompt: str = "",
) -> str:
    """Build the question-plus-notes prompt sent as the first turn of an LLM conversation.

    `corpus_prompt` is the preamble from `build_prompt_context` (date, About me, corpus prompt,
    global prompts) and is placed before the notes; `task_prompt` is the chosen task, placed last.
    """
    from llmware.prompts import PromptCatalog

    context = build_context_from_results(results)
    if not context.strip():
        raise ValueError("No searchable text was available to send to the LLM.")

    query = question.strip()
    if task_prompt.strip():
        query = (f"{query}\n\n" if query else "") + f"Task: {task_prompt.strip()}"

    prompt_dict = PromptCatalog().build_core_prompt(
        prompt_name="default_with_context",
        separator="\n",
        query=query,
        context=context,
        inference_dict={"temperature": temperature, "max_tokens": max_tokens},
    )
    core_prompt = prompt_dict["core_prompt"]
    if corpus_prompt.strip():
        core_prompt = f"{corpus_prompt.strip()}\n\n{core_prompt}"
    return core_prompt


def answer_with_ollama(
    question: str,
    results: list,
    model_name: str,
    host: str = OLLAMA_HOST,
    port: int = OLLAMA_PORT,
    max_tokens: int = 512,
    temperature: float = 0.3,
    corpus_prompt: str = "",
    task_prompt: str = "",
    on_progress=None,
) -> tuple[str, dict]:
    """Generate a natural-language answer from search results using a local Ollama model."""
    full_prompt = build_llm_prompt(question, results, max_tokens, temperature, corpus_prompt, task_prompt)
    return chat_with_ollama(
        [{"role": "user", "content": full_prompt}],
        model_name,
        host=host,
        port=port,
        max_tokens=max_tokens,
        temperature=temperature,
        on_progress=on_progress,
    )


# Cancel floats at the bottom centre of the window, wherever the page is scrolled to.
LLM_CANCEL_STYLE = """<style>
[class*="st-key-llm_cancel_"] {
  position: fixed !important; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
  width: auto !important; z-index: 1000;
}
[class*="st-key-llm_cancel_"] button {
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
  background-color: rgb(255, 140, 0) !important; border-color: rgb(255, 140, 0) !important; color: #fff !important;
}
[class*="st-key-llm_cancel_"] button:hover {
  background-color: rgb(230, 120, 0) !important; border-color: rgb(230, 120, 0) !important;
}
</style>"""
_LLM_CANCEL_KEYS = itertools.count()  # one Cancel per LLM call; restarts each script run


def cancel_llm() -> None:
    st.session_state[LLM_CANCELLED_KEY] = True


def show_llm_cancel_button():
    """A floating orange Cancel Query for the LLM call about to run; returns its slot, to clear when the call ends.

    Clicking it reruns the script, which stops the call at its next page update (see LLM_TICK_SECONDS).
    """
    slot = st.empty()
    number = next(_LLM_CANCEL_KEYS)
    with slot.container(key=f"llm_cancel_{number}"):
        st.html(LLM_CANCEL_STYLE)
        st.button(
            "Cancel Query",
            # Unique per call, as one run can make several (Re-assess); not llm_cancel_*, which the style places.
            key=f"stop_llm_{number}",
            icon=":material/stop_circle:",
            type="primary",
            on_click=cancel_llm,
            help="Stop the LLM now, e.g. to change the task or prompts and run it again.",
        )
    return slot


class OllamaChatStream:
    """POST /api/chat on a worker thread, putting each streamed line on `lines`.

    The script thread stays free to update the page (so Streamlit can act on Cancel), and `close` can cut
    the connection at any point, even while Ollama is still reading the prompt; Ollama stops generating
    when its client goes away.
    """

    def __init__(self, host: str, port: int, payload: dict):
        self.conn = http.client.HTTPConnection(host, port, timeout=10)
        self.conn.connect()  # here, so close() always has a socket to cut
        self.conn.sock.settimeout(900)  # longest wait between streamed pieces: a big prompt can take minutes
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, args=(payload,), daemon=True).start()

    def _run(self, payload: dict) -> None:
        try:
            self.conn.request(
                "POST", "/api/chat", body=json.dumps(payload), headers={"Content-Type": "application/json"}
            )
            response = self.conn.getresponse()
            if response.status != 200:
                self.lines.put(("http_error", (response.status, response.read(500).decode(errors="replace"))))
                return
            for line in response:
                self.lines.put(("line", line))
            self.lines.put(("end", None))
        except Exception as exc:
            self.lines.put(("error", exc))

    def close(self) -> None:
        try:
            self.conn.sock and self.conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()


def chat_with_ollama(
    messages: list[dict],
    model_name: str,
    host: str = OLLAMA_HOST,
    port: int = OLLAMA_PORT,
    max_tokens: int = 512,
    temperature: float = 0.3,
    num_ctx: int | None = None,
    response_format: dict | None = None,
    on_progress=None,
    think: bool | None = None,
) -> tuple[str, dict]:
    """Send a chat conversation to a local Ollama model and return (reply, usage).

    `response_format` is a JSON schema; Ollama then constrains the reply to match it.
    `think=False` asks a reasoning model (e.g. Qwen3) to answer without thinking first; other models ignore it.
    The reply is streamed, so long answers have no overall time limit; `on_progress(tokens, seconds)`
    is called every LLM_PROGRESS_SECONDS with the tokens generated so far and the seconds since the
    first one (None while the model is still reading the prompt).
    A floating Cancel shows while the call runs; clicking it stops the call and closes the connection.
    """
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": context_window_for(messages, max_tokens, num_ctx),
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    if response_format:
        payload["format"] = response_format
    if think is not None:
        payload["think"] = think

    try:
        stream = OllamaChatStream(host, port, payload)
    except OSError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at `{host}:{port}`. "
            f"Start Ollama or run `./start_app.sh`. Details: {exc}"
        ) from exc

    cancel_slot = show_llm_cancel_button()
    tick = st.empty()
    content, thinking, data = [], [], {}
    generated, first_token_at = 0, None
    last_report = time.monotonic()
    try:
        while True:
            try:
                kind, value = stream.lines.get(timeout=LLM_TICK_SECONDS)
            except queue.Empty:
                kind, value = "wait", None
                tick.empty()  # a page update, so a Cancel click takes effect here
            if kind == "http_error":
                status, body = value
                raise RuntimeError(f"Ollama returned HTTP {status} for model `{model_name}`: {body}")
            if kind == "error":
                raise RuntimeError(
                    f"Lost the connection to Ollama while `{model_name}` was answering: {value}"
                ) from value
            if kind == "end":
                break
            if kind == "line" and value.strip():
                try:
                    chunk = json.loads(value)
                except ValueError as exc:
                    raise RuntimeError(f"Ollama returned invalid JSON: {value[:500]!r}") from exc
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error for model `{model_name}`: {chunk['error']}")
                piece = chunk.get("message") or {}
                if piece.get("content") or piece.get("thinking"):
                    content.append(piece.get("content") or "")
                    thinking.append(piece.get("thinking") or "")
                    generated += 1  # Ollama streams about one token per piece
                    first_token_at = first_token_at or time.monotonic()
                if chunk.get("done"):
                    data = chunk
                    break
            if on_progress and time.monotonic() - last_report >= LLM_PROGRESS_SECONDS:
                last_report = time.monotonic()
                on_progress(generated, last_report - first_token_at if first_token_at else None)
    finally:
        # Also runs when Cancel stops the script: dropping the connection stops Ollama generating.
        stream.close()
        cancel_slot.empty()

    message = {"content": "".join(content), "thinking": "".join(thinking)}
    # Reasoning models (e.g. Qwen3) may inline their thinking; keep only the answer.
    answer = re.sub(r"<think>.*?</think>", "", str(message.get("content", "")), flags=re.DOTALL).strip()
    if not answer and message.get("thinking"):
        raise RuntimeError(
            f"Ollama model `{model_name}` used its whole token budget thinking and gave no answer. "
            "Raise **Max tokens** and try again."
        )
    if not answer:
        raise RuntimeError(
            f"Ollama model `{model_name}` returned an empty answer. "
            f"Check that the model is pulled (`ollama pull {model_name}`)."
        )

    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    output_tokens = int(data.get("eval_count") or 0)
    total_duration = data.get("total_duration")
    eval_duration = data.get("eval_duration")
    usage = {
        "input": prompt_tokens,
        "output": output_tokens,
        "total": prompt_tokens + output_tokens,
        "metric": "tokens",
        "processing_time": (total_duration / 1e9) if total_duration else None,
        "tokens_per_second": (output_tokens / (eval_duration / 1e9)) if eval_duration else None,
    }
    return answer, usage


def context_window_for(messages: list[dict], max_tokens: int, minimum: int | None = None) -> int:
    """A context window that fits the prompt (about 3 characters a token) plus the whole answer."""
    needed = sum(len(str(m.get("content", ""))) for m in messages) // 3 + max_tokens + 256
    size = next((s for s in OLLAMA_CONTEXT_SIZES if s >= needed), OLLAMA_CONTEXT_SIZES[-1])
    return max(size, minimum or 0)


def tokens_as_words(tokens: int) -> int:
    return round(tokens * WORDS_PER_TOKEN)


def approximately(number: int) -> str:
    """'≈1,600', or to the nearest thousand for big numbers: '≈127,000'."""
    return f"≈{round(number, -3) if number >= 10000 else number:,}"


def book_token_markers() -> dict[int, str]:
    """{token count: book title} for the books that fit under the Max tokens limit."""
    return {round(words / WORDS_PER_TOKEN): title for title, words in BOOK_LENGTHS}


def max_tokens_options(current: int) -> list[int]:
    """Slider stops: round steps plus the book markers (and the saved value, if it's neither)."""
    return sorted({*TOKEN_STEPS, *book_token_markers(), min(max(int(current), TOKEN_STEPS[0]), LLM_MAX_TOKENS_LIMIT)})


def format_max_tokens(tokens: int) -> str:
    """Slider label: the book at a marker, otherwise the length in words."""
    book = book_token_markers().get(tokens)
    return f"{tokens:,} · {book}" if book else f"{tokens:,} · ≈{tokens_as_words(tokens):,} words"


def describe_answer_length(tokens: int) -> str:
    """How long an answer of `tokens` could be, compared with well-known books."""
    words = tokens_as_words(tokens)
    same = [title for title, book_words in BOOK_LENGTHS if abs(book_words - words) <= book_words * 0.01]
    shorter = [(title, n) for title, n in BOOK_LENGTHS if n < words and title not in same]
    longer = [(title, n) for title, n in BOOK_LENGTHS if n > words and title not in same]
    text = f"Up to {approximately(words)} words"
    if same:
        text += f": about the length of *{same[0]}*"
    else:
        if shorter:
            text += f": longer than *{shorter[-1][0]}* ({approximately(shorter[-1][1])})"
        if longer:
            text += f"{',' if shorter else ':'} shorter than *{longer[0][0]}* ({approximately(longer[0][1])})"
    beyond = ", ".join(
        f"*{title}* {approximately(round(book_words / WORDS_PER_TOKEN))}" for title, book_words in BOOKS_BEYOND_LIMIT
    )
    return text + f". For scale, in tokens: {beyond}."


LLM_SPEED_KEY = "llm_speed"  # app_settings: {model: {"tokens_per_second", "tokens", "at"}} from the last answer
SPEED_BENCHMARK_TOKENS = 10000


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def record_llm_speed(model: str, usage: dict | None) -> None:
    """Keep the generation speed of the model's latest answer, so the next run can show what to expect."""
    tokens_per_second = (usage or {}).get("tokens_per_second")
    if not tokens_per_second:
        return
    speeds = dict(load_app_settings().get(LLM_SPEED_KEY) or {})
    speeds[model] = {
        "tokens_per_second": round(tokens_per_second, 1),
        "tokens": (usage or {}).get("output"),
        "at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    save_app_settings({LLM_SPEED_KEY: speeds})


def last_llm_speed(model: str) -> dict | None:
    return (load_app_settings().get(LLM_SPEED_KEY) or {}).get(model)


def describe_llm_speed(model: str) -> str:
    """The model's last-run speed and the time it implies for 10,000 tokens, shown under Search."""
    record = last_llm_speed(model)
    if not record:
        return f"No speed measured yet for `{model}`; it's recorded after its next answer."
    tokens_per_second = record["tokens_per_second"]
    when = datetime.datetime.fromisoformat(record["at"]).strftime("%d %b %H:%M")
    return (
        f"Last run with `{model}` ({when}): {tokens_per_second:.1f} tokens/s · "
        f"**{SPEED_BENCHMARK_TOKENS:,} tokens ≈ {format_duration(SPEED_BENCHMARK_TOKENS / tokens_per_second)}**"
    )


@contextmanager
def llm_progress(slot, label: str):
    """A spinner in `slot` (under the Search button) with a tokens-per-second count beside it.

    Yields the `on_progress` callback for chat_with_ollama; the slot is cleared afterwards.
    """
    with slot.container():
        spin_col, rate_col = st.columns([3, 2], vertical_alignment="center")
        rate = rate_col.empty()
        rate.caption("Reading the notes…")

        def report(tokens: int, seconds: float | None) -> None:
            if seconds:
                rate.caption(f"**{tokens / seconds:.1f} tokens/s** · {tokens:,} tokens")
            else:
                rate.caption("Reading the notes…")

        with spin_col, st.spinner(label, show_time=True):
            yield report
    slot.empty()


def normalize_text_search_query(query: str) -> str | None:
    """Collapse whitespace and drop punctuation so LLMWare FTS gets clean tokens."""
    cleaned = re.sub(r"[^\w\s]", " ", query or "")
    tokens = [tok for tok in cleaned.split() if tok]
    return " ".join(tokens) if tokens else None


def safe_text_query(q: Query, query: str, result_count: int = 20) -> list:
    """Run a keyword fallback query, skipping empty or punctuation-only input."""
    normalized = normalize_text_search_query(query)
    if not normalized:
        return []
    results = q.text_query(normalized, result_count=result_count)
    return results or []


SEARCH_FILTER_FIELDS = ("all", "any", "none")
ALL_LIBRARIES_KEY = "__all_libraries__"


def clean_filter_terms(terms) -> list[str]:
    """Strip blanks and case-insensitive duplicates, keeping the order terms were added."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms or []:
        text = str(term).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            cleaned.append(text)
    return cleaned


def is_filter_active(search_filter: dict) -> bool:
    return any(search_filter.get(field) for field in SEARCH_FILTER_FIELDS)


def has_positive_filter_terms(search_filter: dict) -> bool:
    return bool(search_filter.get("all") or search_filter.get("any"))


def describe_search_filter(search_filter: dict) -> str:
    """Readable form of the filter, e.g. `2026-10` AND `Acme` AND (`PoC` OR `pilot`) AND NOT `draft`."""
    parts = [f"`{term}`" for term in search_filter.get("all", [])]
    any_terms = search_filter.get("any", [])
    if len(any_terms) == 1:
        parts.append(f"`{any_terms[0]}`")
    elif any_terms:
        parts.append("(" + " OR ".join(f"`{term}`" for term in any_terms) + ")")
    parts.extend(f"NOT `{term}`" for term in search_filter.get("none", []))
    if actioned := describe_actioned_filter():
        parts.append(actioned)
    return " AND ".join(parts)


def build_note_filter_sql(search_filter: dict) -> tuple[str, list]:
    """HAVING clause matching whole notes (any block, the file name or its folder path), case-insensitively.

    Folder paths aren't in the index, so `with_path_matches` lists the notes whose folder matches each term
    under `path_sources`.
    """
    term_in_note = (
        "MAX(instr(lower(coalesce(text_block, '')), ?) > 0 "
        "OR instr(lower(coalesce(file_source, '')), ?) > 0)"
    )
    path_sources = search_filter.get("path_sources") or {}
    params: list = []

    def matches(term: str) -> str:
        params.extend([term.lower(), term.lower()])
        sources = path_sources.get(term.lower(), [])
        if not sources:
            return f"{term_in_note} = 1"
        params.extend(sources)
        return f"({term_in_note} = 1 OR file_source IN ({', '.join('?' for _ in sources)}))"

    clauses: list[str] = []
    for term in search_filter.get("all", []):
        clauses.append(matches(term))
    any_terms = search_filter.get("any", [])
    if any_terms:
        clauses.append("(" + " OR ".join(matches(term) for term in any_terms) + ")")
    for term in search_filter.get("none", []):
        clauses.append(f"NOT {matches(term)}")
    included = search_filter.get("include_sources")
    if included is not None:  # Search after Find notes: only the notes found
        clauses.append(f"file_source IN ({', '.join('?' for _ in included)})" if included else "0")
        params += list(included)
    excluded = search_filter.get("exclude_sources") or []
    if excluded:
        clauses.append(f"file_source NOT IN ({', '.join('?' for _ in excluded)})")
        params += list(excluded)
    return " AND ".join(clauses), params


EXCLUDED_NOTES_KEY = "excluded_notes"


def excluded_notes_state() -> dict[str, list[str]]:
    """{library key: [file_source]} of notes excluded from Search, loaded from settings once per session."""
    if EXCLUDED_NOTES_KEY not in st.session_state:
        saved = load_app_settings().get(EXCLUDED_NOTES_KEY) or {}
        st.session_state[EXCLUDED_NOTES_KEY] = {lib: list(sources) for lib, sources in saved.items() if sources}
    return st.session_state[EXCLUDED_NOTES_KEY]


def get_excluded_notes(lib_key: str) -> list[str]:
    """Notes the user ticked to exclude from Search in this library."""
    return sorted(excluded_notes_state().get(lib_key, []))


def count_excluded_notes() -> int:
    return sum(len(sources) for sources in excluded_notes_state().values())


def with_exclusions(search_filter: dict, lib_key: str) -> dict:
    return with_actioned_filter({**search_filter, "exclude_sources": get_excluded_notes(lib_key)}, lib_key)


def exclude_checkbox_key(lib_key: str, file_source: str) -> str:
    return f"exclude::{lib_key}::{file_source}"


TARGET_PICKS_KEY = "target_picks"
NOTE_SELECTION_HELP = "Check the box to exclude, click the target to select."


def persist_excluded_notes() -> None:
    state = {lib: sources for lib, sources in excluded_notes_state().items() if sources}
    save_app_settings({EXCLUDED_NOTES_KEY: state})


def _set_note_excluded(lib_key: str, file_source: str, excluded: bool) -> None:
    state = excluded_notes_state()
    sources = set(state.get(lib_key, []))
    if excluded:
        sources.add(file_source)
    else:
        sources.discard(file_source)
    state[lib_key] = sorted(sources)
    st.session_state[exclude_checkbox_key(lib_key, file_source)] = excluded


def toggle_excluded_note(lib_key: str, file_source: str) -> None:
    _set_note_excluded(lib_key, file_source, bool(st.session_state.get(exclude_checkbox_key(lib_key, file_source))))
    # A manual change ends any 🎯 run, so the next 🎯 starts afresh.
    st.session_state.pop(TARGET_PICKS_KEY, None)
    persist_excluded_notes()


def target_note(lib_key: str, file_source: str, notes: list[tuple[str, str]]) -> None:
    """🎯 shortcut: include this note and exclude every listed note not also picked with 🎯."""
    target_notes([(lib_key, file_source)], notes)


def target_notes(picked: list[tuple[str, str]], notes: list[tuple[str, str]]) -> None:
    """🎯 on a note or a folder: include these and exclude every listed note not also picked with 🎯."""
    picks = st.session_state.setdefault(TARGET_PICKS_KEY, set())
    picks.update(picked)
    for lib, source in notes:
        _set_note_excluded(lib, source, (lib, source) not in picks)
    persist_excluded_notes()


def toggle_folder_excluded(widget_key: str, notes: list[tuple[str, str]]) -> None:
    set_notes_excluded(notes, bool(st.session_state.get(widget_key)))


def set_notes_excluded(notes: list[tuple[str, str]], excluded: bool) -> None:
    """Tick or untick the exclude box for every listed (library key, file) note."""
    for lib_key, file_source in notes:
        _set_note_excluded(lib_key, file_source, excluded)
    st.session_state.pop(TARGET_PICKS_KEY, None)
    persist_excluded_notes()


def toggle_all_notes_excluded(notes: list[tuple[str, str]]) -> None:
    set_notes_excluded(notes, bool(st.session_state.get("exclude_all_notes")))


def clear_excluded_notes() -> None:
    st.session_state[EXCLUDED_NOTES_KEY] = {}
    st.session_state.pop(TARGET_PICKS_KEY, None)
    persist_excluded_notes()
    for key in [key for key in st.session_state if str(key).startswith("exclude::")]:
        del st.session_state[key]


def find_matching_sources(safe_lib: str, search_filter: dict, limit: int | None = None) -> list[str]:
    """File sources of notes matching the filter, in index order. Empty if the library has no index."""
    having, params = build_note_filter_sql(with_actioned_filter(search_filter, safe_lib))
    if not having:
        return []
    table = safe_lib.replace('"', '""')
    sql = f'SELECT file_source FROM "{table}" GROUP BY file_source HAVING {having} ORDER BY MIN(doc_ID)'
    if limit:
        sql += " LIMIT ?"
        params = params + [limit]
    try:
        with sqlite3.connect(SQLiteConfig.get_uri_string()) as conn:
            return [row[0] for row in conn.execute(sql, params)]
    except sqlite3.Error:
        return []


def load_note_blocks(safe_lib: str, sources: list[str]) -> list[dict]:
    """Every block of the given notes, in document order."""
    if not sources:
        return []
    table = safe_lib.replace('"', '""')
    placeholders = ", ".join("?" for _ in sources)
    with sqlite3.connect(SQLiteConfig.get_uri_string()) as conn:
        rows = conn.execute(
            f'SELECT doc_ID, block_ID, file_source, text_block FROM "{table}" '
            f"WHERE file_source IN ({placeholders}) ORDER BY doc_ID, block_ID",
            sources,
        ).fetchall()
    return [
        {"doc_ID": doc_id, "block_ID": block_id, "file_source": source, "text": text or ""}
        for doc_id, block_id, source, text in rows
    ]


def exact_text_search(safe_lib: str, search_filter: dict, max_documents: int) -> list[dict]:
    """Notes matching the filter terms literally, skipping vectors.

    Returns every block of up to `max_documents` matching notes, in document order.
    """
    return load_note_blocks(safe_lib, find_matching_sources(safe_lib, search_filter, max_documents))


def exact_text_search_all_libraries(
    libraries: list[str],
    search_filter: dict,
    max_documents: int,
    corpus_root: Path,
    apply_exclusions: bool = False,
) -> tuple[list[dict], list[tuple[str, int]]]:
    """Literal filter search across libraries; no embeddings are used.

    Returns (blocks of up to `max_documents` notes, [(library, matching note count)]),
    taking notes from the libraries with the most matches first. With `apply_exclusions`,
    notes the user excluded are left out.
    """
    counts: list[tuple[str, str, int, dict]] = []
    for library in libraries:
        lib_key = resolve_library_registry_name(library)
        lib_filter = with_exclusions(search_filter, lib_key) if apply_exclusions else search_filter
        lib_filter = with_path_matches(
            lib_filter, lib_key, resolve_library_content_path(corpus_root, library), corpus_root
        )
        matches = len(find_matching_sources(lib_key, lib_filter))
        if matches:
            counts.append((library, lib_key, matches, lib_filter))
    counts.sort(key=lambda item: -item[2])

    results: list[dict] = []
    remaining = max_documents
    for library, lib_key, _matches, lib_filter in counts:
        if remaining <= 0:
            break
        blocks = exact_text_search(lib_key, lib_filter, remaining)
        try:
            uploads_dirs = get_library_uploads_dirs(Library().load_library(lib_key))
        except Exception:
            uploads_dirs = ()
        library_folder = resolve_library_content_path(corpus_root, library)
        for block in blocks:
            block.update(
                _library=library,
                _safe_lib=lib_key,
                _library_folder=library_folder,
                _uploads_dirs=uploads_dirs,
            )
        results.extend(blocks)
        remaining -= len({block["file_source"] for block in blocks})

    return results, [(library, matches) for library, _lib_key, matches, _filter in counts]


def filtered_semantic_search(
    q: Query,
    safe_lib: str,
    embedding_model: str,
    prompt_text: str,
    search_filter: dict,
    result_limit: int,
    distance_threshold: float,
) -> tuple[list, str]:
    """Vector search restricted to the blocks of notes that match the filter."""
    import numpy as np

    having, params = build_note_filter_sql(search_filter)
    table = safe_lib.replace('"', '""')
    with sqlite3.connect(SQLiteConfig.get_uri_string()) as conn:
        rowids = [
            int(row[0])
            for row in conn.execute(
                f'SELECT rowid FROM "{table}" WHERE file_source IN '
                f'(SELECT file_source FROM "{table}" GROUP BY file_source HAVING {having})',
                params,
            )
        ]
    if not rowids:
        return [], f"No notes match {describe_search_filter(search_filter) or 'among the notes you found and kept'}."

    q.load_embedding_model()
    query_vector = np.asarray(q.embedding_model.embedding(prompt_text), dtype=float).reshape(-1)
    vector_table = get_lancedb_connection().open_table(get_collection_name(safe_lib, embedding_model))
    # LanceDB ids are the SQLite rowids of the blocks, stored as strings.
    id_list = ", ".join(f"'{rowid}'" for rowid in rowids)
    hits = (
        vector_table.search(query_vector.tolist())
        .where(f"id IN ({id_list})", prefilter=True)
        .select(["id"])
        .limit(result_limit)
        .to_list()
    )
    if not hits:
        return [], "The matching notes have no embeddings yet — run **Rescan** in the left panel."

    distances = {int(hit["id"]): float(hit["_distance"]) for hit in hits}
    placeholders = ", ".join("?" for _ in distances)
    with sqlite3.connect(SQLiteConfig.get_uri_string()) as conn:
        rows = conn.execute(
            f'SELECT rowid, doc_ID, block_ID, file_source, text_block FROM "{table}" '
            f"WHERE rowid IN ({placeholders})",
            list(distances),
        ).fetchall()
    results = sorted(
        (
            {
                "doc_ID": doc_id,
                "block_ID": block_id,
                "file_source": source,
                "text": text or "",
                "distance": distances[rowid],
            }
            for rowid, doc_id, block_id, source, text in rows
        ),
        key=lambda result: result["distance"],
    )

    within = [result for result in results if result["distance"] < distance_threshold]
    if within:
        return within, ""
    return results, (
        f"None of the filtered passages were within the distance threshold ({distance_threshold:.2f}) "
        "— showing the closest ones from the matching notes."
    )


GATHERED_KEY = "gathered_notes"  # notes listed by Find notes (semantic) and Filter search only (literal), built up


def find_notes(
    search_lib: Library,
    query: str,
    search_filter: dict,
    embedding_model: str,
    max_notes: int,
    distance_threshold: float,
) -> tuple[list, str, float | None]:
    """Semantic route's Find notes: the passages of up to `max_notes` notes closest in meaning to `query`."""
    passages, notice, nearest = perform_search(
        search_lib, query, search_filter, embedding_model, max_notes * 4, distance_threshold
    )
    notes: list[str] = []
    for passage in passages:  # closest first
        source = str(passage.get("file_source", ""))
        if source not in notes and len(notes) < max_notes:
            notes.append(source)
    return [p for p in passages if str(p.get("file_source", "")) in notes], notice, nearest


def gathered_notes(safe_lib: str) -> dict | None:
    """The notes gathered for this library: {"library", "seq", "notes", "library_counts"}, or None.

    `notes` maps (library key, file source) to {"route", "seq", "blocks"}, oldest hit first; a note found
    again moves to the end and takes the route (semantic or literal) of that search.
    """
    gathered = st.session_state.get(GATHERED_KEY)
    return gathered if gathered and gathered["library"] == safe_lib and gathered["notes"] else None


def gather_results(
    safe_lib: str, results: list, route: str, library_counts: list | None = None, uploads_dirs: tuple = ()
) -> None:
    """Add a listing's notes to the gathered notes, marked with the route that found them."""
    gathered = st.session_state.get(GATHERED_KEY)
    if not gathered or gathered["library"] != safe_lib:
        gathered = {"library": safe_lib, "seq": 0, "notes": {}, "library_counts": []}
    gathered["seq"] += 1
    by_note: dict[tuple[str, str], list[dict]] = {}
    for block in results:
        block.setdefault("_uploads_dirs", uploads_dirs)  # where its attachments are, for the note's preview
        by_note.setdefault((block.get("_safe_lib", safe_lib), str(block.get("file_source", ""))), []).append(block)
    for key, blocks in by_note.items():
        gathered["notes"].pop(key, None)
        gathered["notes"][key] = {"route": route, "seq": gathered["seq"], "blocks": blocks}
    if library_counts:
        gathered["library_counts"] = library_counts
    st.session_state[GATHERED_KEY] = gathered


def restrict_gathered_to_library(safe_lib: str) -> None:
    """Drop gathered notes from other libraries and their per-library counts.

    For a literal search with Search all libraries unticked: its output is this library only, even when
    an earlier all-library search gathered notes elsewhere.
    """
    gathered = st.session_state.get(GATHERED_KEY)
    if not gathered or gathered["library"] != safe_lib:
        return
    gathered["notes"] = {key: entry for key, entry in gathered["notes"].items() if key[0] == safe_lib}
    gathered["library_counts"] = []
    st.session_state[GATHERED_KEY] = gathered


def kept_gathered_notes(gathered: dict) -> list[tuple[str, str]]:
    """Gathered notes not excluded with 🚫, most recent hit first."""
    return [key for key in reversed(gathered["notes"]) if key[1] not in get_excluded_notes(key[0])]


def gathered_full_notes(gathered: dict, keys: list[tuple[str, str]]) -> list[dict]:
    """Every block of these gathered notes: literal listings hold whole notes, semantic ones only passages."""
    blocks: list[dict] = []
    for key in keys:
        entry = gathered["notes"][key]
        note_blocks = entry["blocks"] if entry["route"] == "literal" else load_note_blocks(key[0], [key[1]])
        blocks += [{**block, "_safe_lib": key[0]} for block in note_blocks]  # its library, for the actions log
    return blocks


def literal_matches(
    search_filter: dict, safe_lib: str, library_folder: Path, corpus_root: Path | None, libraries: list[str] | None
) -> set[tuple[str, str]]:
    """(library key, file) of every note the filter terms match now, minus excluded ones: what Ask (filtered
    only) answers from (up to Max results). `libraries`: all of them, when Search all libraries is ticked."""
    targets = ([(resolve_library_registry_name(lib), resolve_library_content_path(corpus_root, lib)) for lib in libraries]
               if libraries and corpus_root else [(safe_lib, library_folder)])
    # Each library's filter carries its excluded notes, so the cache below sees exclusions change.
    return set(_literal_matches(tuple(
        (key, str(folder), json.dumps(with_exclusions(search_filter, key), sort_keys=True)) for key, folder in targets
    ), str(corpus_root or "")))


@st.cache_data(ttl=20, show_spinner=False)
def _literal_matches(targets: tuple, corpus_root: str) -> list[tuple[str, str]]:
    found = []
    for lib_key, folder, filter_json in targets:
        lib_filter = with_path_matches(json.loads(filter_json), lib_key, Path(folder), Path(corpus_root) if corpus_root else None)
        found += [(lib_key, source) for source in find_matching_sources(lib_key, lib_filter)]
    return found


def share_context(gathered: dict, keys: list[tuple[str, str]], budget: int) -> tuple[list[dict], int]:
    """The blocks of these in-scope notes, each note cut to an equal share of `budget` characters, so all of
    them reach the LLM. Returns (blocks, how many notes were trimmed)."""
    share = max(budget // max(len(keys), 1) - 120, 400)  # 120: each note's source line in the context
    blocks, trimmed = [], 0
    for key in keys:
        used = 0
        for block in gathered_full_notes(gathered, [key]):
            text = str(block.get("text", ""))
            if used >= share:
                trimmed += 1
                break
            if used + len(text) > share:
                blocks.append({**block, "text": text[:share - used].rstrip() + " […]"})
                trimmed += 1
                break
            blocks.append(block)
            used += len(text)
    return blocks, trimmed


def clear_gathered_notes() -> None:
    st.session_state.pop(GATHERED_KEY, None)


def clear_search_results() -> None:
    """Clear results button: the gathered notes and the last search's results (and its discussion)."""
    clear_gathered_notes()
    st.session_state.pop(LAST_SEARCH_KEY, None)


def clear_all_search_page() -> None:
    """Clear all button: filter, exclusions, results and discussion, the question, Find notes about, the
    injected task prompts and the Ask box's Salesforce chip. Settings (route, Max results, model) stay."""
    clear_search_filter()
    clear_search_results()
    for field in SEARCH_FILTER_FIELDS:
        st.session_state[f"searched_dates_{field}"] = []
    st.session_state["question_text"] = ""
    persist_question()
    st.session_state["find_notes_query"] = ""
    st.session_state[ACTIVE_TASK_PROMPTS_KEY] = []
    st.session_state.pop("question_task_token", None)
    save_app_settings({SALESFORCE_CONTEXT_SETTING: False})
    for key in [key for key in st.session_state if str(key).endswith("_follow_up_text")]:
        del st.session_state[key]


def perform_search(
    search_lib: Library,
    prompt_text: str,
    search_filter: dict,
    embedding_model: str,
    result_limit: int,
    distance_threshold: float,
) -> tuple[list, str, float | None]:
    """Run the search pipeline.

    Returns (result blocks, notice for the user, closest vector distance when no result carries one).
    """
    search_filter = with_actioned_filter(search_filter, search_lib.library_name)
    q = Query(
        search_lib,
        embedding_model_name=embedding_model,
        vector_db=VECTOR_DB,
    )

    if is_filter_active(search_filter) or search_filter.get("include_sources") is not None:
        results, notice = filtered_semantic_search(
            q,
            search_lib.library_name,
            embedding_model,
            prompt_text,
            search_filter,
            result_limit,
            distance_threshold,
        )
        return results, notice, None

    excluded = set(search_filter.get("exclude_sources") or [])

    def drop_excluded(blocks: list) -> list:
        return [block for block in blocks or [] if block.get("file_source") not in excluded]

    results = drop_excluded(q.semantic_query(
        prompt_text,
        result_count=result_limit + len(excluded) * 3,
        embedding_distance_threshold=distance_threshold,
    ))
    if results:
        return results[:result_limit], "", None

    # Re-run without a cutoff to report how close the best vector match was.
    nearest_distance = None
    nearest = q.semantic_query(prompt_text, result_count=1, embedding_distance_threshold=1e9)
    if nearest and nearest[0].get("distance") is not None:
        nearest_distance = float(nearest[0]["distance"])
    nearest_hint = describe_nearest_vector_match(nearest_distance, distance_threshold)

    results = drop_excluded(safe_text_query(q, prompt_text, result_count=result_limit + len(excluded) * 3))
    if results:
        return results[:result_limit], (
            "Vector search had no matches — showing text search results instead." + nearest_hint
        ), nearest_distance
    return [], "No matches found." + nearest_hint, nearest_distance


def describe_nearest_vector_match(nearest_distance: float | None, distance_threshold: float) -> str:
    """Explain how far the closest vector match was from the distance threshold."""
    if nearest_distance is None:
        return ""
    return (
        f" Closest vector match was **{nearest_distance:.3f}** "
        f"(cutoff {distance_threshold:.2f}). Raise **Distance cutoff** to "
        f"**{cutoff_to_include(nearest_distance):.2f}** to include it — see the amber band under the slider."
    )


_migrated_tables = migrate_all_legacy_vectors()

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
# Each sub-page names its purpose in the browser tab.
PAGE_TITLES = {
    "prompts": "RAG - Prompt library",
    "actions": "RAG - Actions",
    "configure": "RAG - Configure",
    "reference": "RAG - Reference document",
}
st.set_page_config(page_title=PAGE_TITLES.get(st.query_params.get("page"), "RAG Workspace Manager"),
                   layout="wide")
# Starts the background auto-rescan thread once per server process.
get_text_rescan_scheduler()
if st.session_state.pop(LLM_CANCELLED_KEY, False):
    st.toast("Stopped the LLM. Change what you need and run it again.", icon=":material/stop_circle:")
if st.query_params.get("page") == "prompts":
    render_prompt_library_page()
    st.stop()
if st.query_params.get("page") == "actions":
    render_actions_page()
    st.stop()
if st.query_params.get("page") == "configure":
    render_configure_page()
    st.stop()
if st.query_params.get("page") == "reference":
    render_reference_page()
    st.stop()

render_outbox_status()
st.title("🧠 RAG Workspace Manager")

if should_show_setup_wizard():
    render_setup_wizard()
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR — CORPUS ROOT
# ---------------------------------------------------------------------------

st.sidebar.header("Notes location")
if st.sidebar.button("Browse…", key="browse_corpus_root", use_container_width=True):
    current = load_corpus_root_path()
    picked = browse_for_directory(str(current) if current else None)
    if picked:
        picked = picked.strip()
        save_app_settings({"corpus_root_path": picked})
        st.session_state["corpus_root_path_input"] = picked
        st.rerun()

sync_corpus_root_widget_from_settings()
_saved_corpus_root = load_corpus_root_path()
st.sidebar.text_input(
    "Corpus root folder",
    value=str(_saved_corpus_root) if _saved_corpus_root else "",
    key="corpus_root_path_input",
    on_change=persist_corpus_root_path,
    help="Notes export folder. Multi-Corpus: one subfolder per library. Single-Corpus: index the whole tree.",
    placeholder="Select or enter your notes export folder…",
)

corpus_root_path = get_corpus_root_path()
if not corpus_root_path:
    st.sidebar.info("Set the corpus root folder above to load your libraries.")
    st.stop()
if not corpus_root_path.exists():
    st.sidebar.warning(f"Folder not found:\n`{corpus_root_path}`")
    st.stop()

st.sidebar.caption(f"Saved in `{APP_SETTINGS_PATH.name}`")

sync_corpus_mode_from_settings()
st.sidebar.radio(
    "Corpus mode",
    options=[CORPUS_MODE_MULTI, CORPUS_MODE_SINGLE],
    format_func=lambda mode: "Multi-Corpus" if mode == CORPUS_MODE_MULTI else "Single-Corpus",
    key="corpus_mode_toggle",
    horizontal=True,
    on_change=persist_corpus_mode,
    help=(
        "**Multi-Corpus** — each immediate subfolder is a separate searchable library. "
        "**Single-Corpus** — one library for all notes under the root (flat or mixed layouts)."
    ),
)
if is_single_corpus_mode():
    st.sidebar.caption("Single-Corpus indexes every supported file under the root, including subfolders.")
else:
    st.sidebar.caption("Multi-Corpus treats each top-level subfolder as its own library.")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# SIDEBAR — LIBRARY MANAGEMENT
# ---------------------------------------------------------------------------

st.sidebar.header("Library Controls")
if _migrated_legacy_app_data:
    st.sidebar.info(
        "Copied previous data from `~/llmware_data` into `./llmware_data` "
        f"({', '.join(_migrated_legacy_app_data)})."
    )
if _migrated_tables:
    st.sidebar.info(
        f"Migrated {len(_migrated_tables)} vector index(es) from `/tmp/lancedb`."
    )

all_libraries = get_libraries()
if not all_libraries:
    if is_single_corpus_mode():
        st.sidebar.error(
            f"No ingestible files found under:\n`{corpus_root_path}`\n\n"
            f"Supported types: {', '.join(sorted(INGEST_EXTENSIONS))}"
        )
    else:
        st.sidebar.error(f"No subfolders found in:\n`{corpus_root_path}`")
    st.stop()
# All-corpora scans and rebuilds skip hidden corpora.
scan_libraries = filter_libraries_for_display(all_libraries)
# Agents' actions, brought up to date (receipts, and marks in notes rescanned since) for the badges and filter.
sync_actions_log([resolve_library_registry_name(lib) for lib in scan_libraries])

if is_single_corpus_mode():
    selected_lib = SINGLE_CORPUS_DISPLAY_NAME
    st.sidebar.markdown(f"**Library:** {SINGLE_CORPUS_DISPLAY_NAME}")
else:
    visible_libraries = build_library_select_options(all_libraries)
    if not visible_libraries:
        st.sidebar.warning("All corpora are hidden. Turn off hiding in Advanced or unhide a corpus.")
        visible_libraries = all_libraries

    sync_selected_library_from_settings(visible_libraries)
    selected_lib = st.sidebar.selectbox(
        "Select Library",
        visible_libraries,
        key="selected_library",
        on_change=persist_selected_library,
    )
    if is_hide_corpora_enabled():
        hidden_count = len(all_libraries) - len(filter_libraries_for_display(all_libraries))
        if hidden_count:
            st.sidebar.caption(f"{hidden_count} corpus/corpora hidden from this list.")

    if selected_lib != load_last_selected_library():
        save_app_settings({"last_selected_library": selected_lib})

safe_lib = resolve_library_registry_name(selected_lib)
library_content_path = resolve_library_content_path(corpus_root_path, selected_lib)
corpus_config = get_corpus_config(safe_lib)
render_prompt_library_sidebar(corpus_config)

model_catalog = list(load_embedding_model_catalog())
model_ids = [model["id"] for model in model_catalog]
model_labels = {model["id"]: model["label"] for model in model_catalog}

# Status display
try:
    _status_lib = Library().load_library(safe_lib)
    active_model = get_active_library_embedding(_status_lib)
    status_model = get_pending_index_settings(safe_lib, corpus_config)[0]
    emb_ready, emb_message = is_embedding_ready(_status_lib, status_model)
    if emb_ready:
        st.sidebar.success(f"**Status:** Ready — {emb_message}")
    else:
        st.sidebar.warning(f"**Status:** Not ready — {emb_message}")
except Exception:
    _status_lib = None
    active_model = None
    st.sidebar.warning(
        "Library not yet initialised. Use **Re-index (full rebuild)** below to create the search index."
    )

last_updated = get_last_updated(library_content_path)
if last_updated:
    st.sidebar.caption(f"Last indexed: {last_updated}")

with st.sidebar.expander("Advanced — Indexing & Embeddings", expanded=False):
    _app_settings = load_app_settings()
    st.checkbox(
        "Also use GPU for embeddings",
        value=bool(_app_settings.get("use_gpu_for_embeddings", False)),
        key="use_gpu_for_embeddings_global",
        on_change=persist_gpu_for_embeddings_setting,
        help=(
            "Global setting for all corpus scans. Uses CUDA when available, "
            "or the Apple Silicon GPU (MPS) on Mac. Models marked (g) can use the GPU."
        ),
    )
    auto_rescan_enabled = st.checkbox(
        f"Auto-rescan every {AUTO_RESCAN_SECONDS // 60} min (text only)",
        value=bool(_app_settings.get("auto_rescan_enabled", False)),
        key="auto_rescan_enabled",
        on_change=persist_auto_rescan_settings,
        help=(
            "While the app's server is running (even with no browser tab open), checks the selected "
            "corpus for new, edited and deleted notes and updates the text index, so filters and "
            "literal search find them. Only changed notes are read.\n\n"
            "Embeddings are not built: semantic search sees the changes after **⚡ Rescan**."
        ),
    )
    st.checkbox(
        "…in every corpus",
        value=bool(_app_settings.get("auto_rescan_all_corpora", False)),
        key="auto_rescan_all_corpora",
        on_change=persist_auto_rescan_settings,
        disabled=not auto_rescan_enabled or is_single_corpus_mode(),
        help=(
            "Rescan every visible sub-corpus, not just the selected one — so a note added to another "
            "corpus turns up in **Search all libraries**. Hidden corpora are skipped. Still text only."
        ),
    )
    st.caption(f"Global — saved to `{APP_SETTINGS_PATH}`")
    st.divider()
    st.caption(f"Per-corpus settings for **{selected_lib}**")

    show_all_models = st.checkbox(
        "Show all local embedding models",
        value=False,
        key=f"show_all_models_{safe_lib}",
    )
    visible_model_ids = model_ids if show_all_models else [
        model_id for model_id in RECOMMENDED_EMBEDDING_MODELS if model_id in model_ids
    ]
    if corpus_config["embedding_model"] not in visible_model_ids:
        visible_model_ids = [corpus_config["embedding_model"]] + visible_model_ids

    selected_embedding_model = st.selectbox(
        "Embedding engine (semantic search)",
        options=visible_model_ids,
        index=visible_model_ids.index(corpus_config["embedding_model"]),
        format_func=lambda model_id: format_embedding_model_option(model_id, model_labels),
        key=f"embedding_model_{safe_lib}",
    )
    st.info(get_embedding_model_summary(selected_embedding_model))
    extra_deps_missing = get_missing_embedding_dependencies(selected_embedding_model)
    if extra_deps_missing:
        render_embedding_dependency_install_help(extra_deps_missing, selected_embedding_model)

    if f"distance_threshold_{safe_lib}" not in st.session_state:
        st.session_state[f"distance_threshold_{safe_lib}"] = float(corpus_config["distance_threshold"])
    distance_threshold = st.slider(
        "Relevance cutoff (distance)",
        min_value=DISTANCE_SLIDER_MIN,
        max_value=DISTANCE_SLIDER_MAX,
        step=DISTANCE_SLIDER_STEP,
        help="Search tuning for this corpus. Lower = stricter matches.",
        key=f"distance_threshold_{safe_lib}",
        on_change=sync_cutoff_from_sidebar,
        args=(safe_lib,),
    )

    settings_changed = (
        selected_embedding_model != corpus_config["embedding_model"]
        or abs(distance_threshold - float(corpus_config["distance_threshold"])) > 0.001
    )
    needs_rebuild_for_model = bool(active_model and active_model != selected_embedding_model)
    first_index = is_first_index_run(active_model)
    requires_reembed_ack = not first_index and (
        selected_embedding_model != corpus_config["embedding_model"] or needs_rebuild_for_model
    )

    if requires_reembed_ack:
        st.warning(
            "Changing the embedding engine requires a **Full Hard Rebuild**. "
            "Distance cutoff changes only affect search — no re-embed needed."
        )
    elif not first_index and settings_changed and not needs_rebuild_for_model:
        st.info("Distance cutoff changes only affect search — no re-embed needed.")

    if requires_reembed_ack:
        if selected_embedding_model != corpus_config["embedding_model"]:
            st.checkbox(
                "I understand changing the embedding engine requires a **Full Hard Rebuild**.",
                value=False,
                key=f"reembed_ack_{safe_lib}",
            )
        elif needs_rebuild_for_model:
            st.checkbox(
                "I understand applying the saved embedding engine requires a **Full Hard Rebuild**.",
                value=False,
                key=f"reembed_ack_active_{safe_lib}",
            )

    if st.button("Save corpus settings", key=f"save_settings_{safe_lib}"):
        save_corpus_config(
            safe_lib,
            {
                "embedding_model": selected_embedding_model,
                "distance_threshold": distance_threshold,
                "vector_db": VECTOR_DB,
            },
        )
        st.success("Saved for this corpus.")

    if not is_single_corpus_mode():
        render_corpus_visibility_controls(all_libraries)

        st.divider()
        st.markdown("**Multi-corpus clean rebuild**")
        st.caption(
            f"Full hard rebuild of the {len(scan_libraries)} visible sub-corpora (hidden ones are "
            "skipped), in order, using one embedding engine."
        )
        multi_show_all = st.checkbox(
            "Show all local embedding models",
            value=False,
            key="multi_rebuild_show_all_models",
        )
        multi_model_ids = model_ids if multi_show_all else [
            model_id for model_id in RECOMMENDED_EMBEDDING_MODELS if model_id in model_ids
        ]
        if corpus_config["embedding_model"] not in multi_model_ids:
            multi_model_ids = [corpus_config["embedding_model"]] + multi_model_ids
        multi_rebuild_model = st.selectbox(
            "Embedding engine for all corpora",
            options=multi_model_ids,
            index=multi_model_ids.index(corpus_config["embedding_model"]),
            format_func=lambda model_id: format_embedding_model_option(model_id, model_labels),
            key="multi_rebuild_model",
        )
        multi_deps_missing = get_missing_embedding_dependencies(multi_rebuild_model)
        if multi_deps_missing:
            render_embedding_dependency_install_help(multi_deps_missing, multi_rebuild_model)
        st.caption("Order: " + " → ".join(scan_libraries))
        multi_rebuild_ack = st.checkbox(
            "I understand this deletes and rebuilds the index for **every** visible sub-corpus.",
            value=False,
            key="multi_rebuild_ack",
        )
        if st.button(
            "🔨 Rebuild all corpora",
            disabled=not multi_rebuild_ack or bool(multi_deps_missing) or not scan_libraries,
            key="multi_rebuild_start",
        ):
            st.session_state["multi_rebuild_request"] = multi_rebuild_model
        if st.button(
            "🔨 Rebuild all corpora — text only",
            disabled=not multi_rebuild_ack or not scan_libraries,
            key="multi_rebuild_text_only_start",
            help=(
                "Deletes every visible sub-corpus's index (hidden ones are skipped), **embeddings included**, and rebuilds only the text, "
                "so filters and literal search work again straight away. Semantic search in a corpus then "
                "needs **⚡ Rescan**, which re-embeds every note in it.\n\n"
                "Rescans (auto or manual) already pick up new, edited and deleted notes, so you rarely "
                "need this — use it when the text index seems out of step with your notes, for example "
                "after a scan was interrupted."
            ),
        ):
            st.session_state["multi_rebuild_request"] = multi_rebuild_model
            st.session_state["multi_rebuild_text_only"] = True

with st.sidebar.expander("Advanced — Actions log", expanded=False):
    render_actions_log_backup()

with st.sidebar.expander("Debug details", expanded=False):
    render_debug_panel(selected_lib, safe_lib, _status_lib, corpus_config)

st.sidebar.divider()
st.sidebar.markdown(f"**This corpus** — {selected_lib}")

pending_model, pending_threshold, reembed_ack = get_pending_index_settings(safe_lib, corpus_config)
first_index = is_first_index_run(active_model)
model_change_pending = pending_model != corpus_config["embedding_model"] or bool(
    active_model and active_model != pending_model
)
sync_disabled = not first_index and model_change_pending
rebuild_disabled = not first_index and model_change_pending and not reembed_ack

# The two incremental rescans side by side: this corpus, and every visible corpus.
with st.sidebar.container(horizontal=True, gap="small"):
    sync_clicked = st.button("⚡ Rescan (incremental)", disabled=sync_disabled, key=f"sync_{safe_lib}")
    multi_rescan_clicked = not is_single_corpus_mode() and st.button(
        "⚡ Rescan all corpora",
        key="multi_rescan_start",
        disabled=not scan_libraries,
        help=(
            f"Rescans the {len(scan_libraries)} visible sub-corpora (hidden ones are skipped) in turn, like the "
            "Getting started wizard: new and edited notes are indexed and embedded, deleted notes "
            "dropped. Each corpus keeps its own embedding engine and nothing is wiped — only changes "
            "are processed, so after the first run this is quick."
        ),
    )

if sync_clicked:
    try:
        ensure_embedding_dependencies(pending_model)
        ensure_corpus_root_persisted(corpus_root_path)
        set_embedding_acceleration(get_use_gpu_for_embeddings())

        def _sync_incremental() -> dict:
            lib = load_or_create_library(safe_lib, embedding_model=pending_model)
            return sync_library(lib, library_content_path, pending_model)

        job_status = run_with_embedding_progress(
            f"Syncing '{selected_lib}'", safe_lib, pending_model, _sync_incremental
        )
        commit_job_status(job_status, selected_lib, safe_lib)
        save_corpus_config(
            safe_lib,
            {
                "embedding_model": pending_model,
                "distance_threshold": pending_threshold,
                "vector_db": VECTOR_DB,
            },
        )
        st.rerun()
    except Exception as exc:
        handle_embedding_run_error(exc, pending_model, sidebar=True)

if st.sidebar.button("🔨 Re-index (full rebuild)", disabled=rebuild_disabled, key=f"rebuild_{safe_lib}"):
    try:
        ensure_embedding_dependencies(pending_model)
        ensure_corpus_root_persisted(corpus_root_path)
        set_embedding_acceleration(get_use_gpu_for_embeddings())

        def _sync_rebuild() -> dict:
            lib = load_or_create_library(safe_lib, force_create=True, embedding_model=pending_model)
            return sync_library(lib, library_content_path, pending_model)

        job_status = run_with_embedding_progress(
            f"Rebuilding '{selected_lib}'", safe_lib, pending_model, _sync_rebuild
        )
        commit_job_status(job_status, selected_lib, safe_lib)
        save_corpus_config(
            safe_lib,
            {
                "embedding_model": pending_model,
                "distance_threshold": pending_threshold,
                "vector_db": VECTOR_DB,
            },
        )
        st.rerun()
    except Exception as exc:
        handle_embedding_run_error(exc, pending_model, sidebar=True)

if multi_rescan_clicked:
    ensure_corpus_root_persisted(corpus_root_path)
    set_embedding_acceleration(get_use_gpu_for_embeddings())
    st.session_state["multi_rebuild_summary"] = run_multi_corpus_rescan(scan_libraries, corpus_root_path)
    st.rerun()

multi_rebuild_request = st.session_state.pop("multi_rebuild_request", None)
multi_rebuild_text_only = st.session_state.pop("multi_rebuild_text_only", False)
if multi_rebuild_request and not is_single_corpus_mode():
    try:
        if not multi_rebuild_text_only:
            ensure_embedding_dependencies(multi_rebuild_request)
        ensure_corpus_root_persisted(corpus_root_path)
        set_embedding_acceleration(get_use_gpu_for_embeddings())
        st.session_state["multi_rebuild_summary"] = run_multi_corpus_rebuild(
            scan_libraries, corpus_root_path, multi_rebuild_request, text_only=multi_rebuild_text_only
        )
        st.session_state.pop("multi_rebuild_ack", None)
        st.rerun()
    except Exception as exc:
        handle_embedding_run_error(exc, multi_rebuild_request, sidebar=True)

render_multi_corpus_rebuild_summary()
render_sidebar_job_summary(selected_lib, safe_lib)

# Reload saved config for search (may have just been updated in session)
corpus_config = get_corpus_config(safe_lib)
selected_embedding_model, distance_threshold, _ = get_pending_index_settings(safe_lib, corpus_config)
distance_threshold = float(distance_threshold)
library_folder = library_content_path


# ---------------------------------------------------------------------------
# MAIN INTERFACE — SEARCH
# ---------------------------------------------------------------------------
st.header(f"Search: {selected_lib}")
st.caption(
    f"Corpus embedding: **{get_model_label(selected_embedding_model)}** · "
    f"cutoff **{distance_threshold:.2f}** — "
    f"{get_embedding_model_summary(selected_embedding_model)}"
)

# ?pick=<reference document id>: the search page is used to choose that document's notes (no LLM).
pick_ref = find_reference_document(st.query_params.get("pick"))
if st.query_params.get("pick") and pick_ref is None:
    st.warning("That reference document doesn't exist (any more). Close this tab.")
elif pick_ref is not None:
    st.info(f"Choosing notes for the reference document **{md_escape_text(reference_label(pick_ref))}**: "
            "gather notes with the Filter box, then **Add** them in the list under it.")

# Pick mode is one column: Filter box, the document being filled, then the results to add from.
filter_box_col, ask_box_col = (st.container(), st.container()) if pick_ref is not None else st.columns(2, gap="medium")

with filter_box_col:
    with st.container(border=True, key="filter_box"):
        title_col, clear_filter_col, clear_results_col, clear_all_col = st.columns(
            [1.6, 1.2, 1.2, 1], vertical_alignment="center"
        )
        title_col.markdown("**Filter**")
        clear_filter_col.button(
            "Clear Filter",
            on_click=clear_search_filter,
            use_container_width=True,
            help="Remove all filter terms and note exclusions.",
        )
        clear_results_col.button(
            "Clear results",
            on_click=clear_search_results,
            use_container_width=True,
            disabled=not (gathered_notes(safe_lib) or st.session_state.get(LAST_SEARCH_KEY)),
            help="Clear the gathered notes and the last results, so **Search** looks through the whole library "
            "again. Filter terms and exclusions stay.",
        )
        clear_all_col.button(
            "Clear all",
            key="clear_all",
            on_click=clear_all_search_page,
            use_container_width=True,
            help="Start the page afresh: clears the filter, exclusions, results and discussion, your question, "
            "**Find notes about**, any injected task prompt and the **☁️ Salesforce** reference. Settings such "
            "as the route, **Max results** and the model stay.",
        )
        # Two routes into the notes: semantic (vector) search or literal text matching. The controls below
        # follow the route; the Ask box's Search button uses it too.
        if "search_route" not in st.session_state:
            st.session_state["search_route"] = saved_search_route()
        with st.container(key="route_switch"):  # coloured per route by route_tint_style
            st.segmented_control(
                "Search route",
                options=list(SEARCH_ROUTES),
                format_func=SEARCH_ROUTES.get,
                key="search_route",
                on_change=persist_search_route,
                width="stretch",
                help=(
                    "**Semantic** ranks passages by meaning (embeddings); filter terms narrow which notes it "
                    "looks in, and **Distance cutoff** sets how close a match must be.\n\n"
                    "**Literal** returns the notes that match the filter terms, in full. Much faster, works "
                    "across libraries, and the question is only used for the LLM answer."
                ),
            )
        exact_text_only = st.session_state.get("search_route") == "literal"
        st.html(route_tint_style("literal" if exact_text_only else "semantic"))
        find_clicked = False
        find_query = ""
        if not exact_text_only:
            # Semantic route: find, then down-select, then ask. Find lists the notes closest in meaning; untick or 🎯
            # the ones you want, and Search answers from those. Search on its own still does it all in one go.
            with st.container(border=True, gap="xsmall", key="find_notes_box"):
                find_query = st.text_input(
                    "Find notes about",
                    key="find_notes_query",
                    placeholder="e.g. pilot pricing discussions",
                    help=(
                        "Lists the notes closest in meaning to this (within **Narrow by terms**, if set), without an LLM "
                        "answer, and adds them to the results under this box (in blue). Exclude 🚫 or keep 🎯 notes "
                        "there, then **Search** answers your question from the notes you kept. **Max results** is "
                        "the number of notes."
                    ),
                )
        _saved_filter = load_app_settings().get("search_filter") or {}
        FILTER_WIDGETS = {
            "all": (
                "Must include (AND)",
                "Notes must contain **every** one of these terms.",
            ),
            "any": (
                "Any of (OR)",
                "Notes must contain **at least one** of these terms.",
            ),
            "none": (
                "Exclude (NOT)",
                "Notes containing **any** of these terms are left out.",
            ),
        }
        st.html(FILTER_FIELD_STYLE)
        # On the Semantic route the terms only narrow which notes are searched, so they're folded away.
        terms_by_field = {
            field: st.session_state.get(f"filter_{field}", _saved_filter.get(field, [])) for field in SEARCH_FILTER_FIELDS
        }
        set_terms = [t for terms in terms_by_field.values() for t in terms]
        if not exact_text_only:  # a filter set when you start a new search is shown, not folded away
            reveal_filter_for_new_search(terms_by_field)
        terms_area = st.container() if exact_text_only else st.expander(
            "Narrow by terms" + (f" · {len(set_terms)} set ({truncate_text(', '.join(set_terms), 40)})" if set_terms else ""),
            key="narrow_by_terms",
            icon=":material/filter_alt:",
            on_change="rerun",
        )
        with terms_area:
            for field, (label, field_help) in FILTER_WIDGETS.items():
                _key = f"filter_{field}"
                if _key not in st.session_state:
                    st.session_state[_key] = clean_filter_terms(_saved_filter.get(field, []))
                    # Dates restored from the last session count as left over from an earlier search.
                    st.session_state[f"searched_dates_{field}"] = date_terms(st.session_state[_key])
                # Each field is one small box: heading and quick dates, quick terms, then the terms themselves.
                # Any of (OR) and Exclude (NOT) are folded, with their terms in the title, to keep the box short.
                folded = field != "all"
                field_terms = st.session_state[_key]
                field_box = st.expander(
                    label + (f" · {len(field_terms)}: {truncate_text(', '.join(field_terms), 40)}" if field_terms else ""),
                    key=f"filter_fold_{field}",
                    on_change="rerun",
                ) if folded else st.container(border=True, gap="xsmall", key=f"filter_field_{field}")
                with field_box:
                    render_filter_heading(field, label, (
                        f"{field_help}\n\n"
                        "Each term is matched **literally** (case-insensitive) anywhere in the note or its "
                        "file name, so `2026-10` matches `2026-10-19` but not `2026 … 10`. "
                        + (
                            "On the **Literal** route, **Search** returns the matching notes in full."
                            if exact_text_only
                            else "On the **Semantic** route, passages are ranked **only within** the matching notes."
                        )
                    ), show_label=not folded)
                    st.multiselect(
                        label,
                        options=st.session_state[_key],
                        key=_key,
                        accept_new_options=True,
                        on_change=persist_search_filter,
                        placeholder="Type a term, press Enter",
                        label_visibility="collapsed",
                    )
        # How many passages (Semantic) or notes (Literal) a search returns: a search setting, so it's here.
        _max_key = max_results_key(exact_text_only)
        _max_limit = MAX_RESULTS_LIMITS["literal" if exact_text_only else "semantic"]
        if _max_key not in st.session_state:  # saved between searches and sessions
            st.session_state[_max_key] = min(max(int(load_app_settings().get(_max_key, 15) or 15), 1), _max_limit)
        search_filter = get_search_filter_from_widgets()
        # One line: Max results (its title beside the box), then what the filter and exclusions leave out, with
        # Clear exclusions rightmost.
        with st.container(horizontal=True, vertical_alignment="center", gap="small", key="max_results_row"):
            st.markdown("Max results", width="content",
                        help="The most passages (Semantic route, up to 50) or notes (Literal route, up to 1000) a "
                        f"search returns. **Ask (filtered only)** answers from at most {ASK_NOTES_LIMIT} notes; "
                        "**Filter search only** lists up to the full number. Each route's value is saved.")
            result_limit = st.number_input(
                "Max results",
                min_value=1,
                max_value=_max_limit,
                key=_max_key,
                on_change=persist_max_results,
                args=(_max_key,),
                width=80,
                label_visibility="collapsed",
            )
            # The filter takes the free space and wraps within it, so the rest stays on this line.
            st.caption(f"Filter: {describe_search_filter(search_filter)}"
                       if is_filter_active(search_filter) or describe_actioned_filter() else "")
            if count_excluded_notes():
                st.caption(f"🚫 {count_excluded_notes()} excluded", width="content")
                st.button("Clear", key="clear_exclusions", type="tertiary", on_click=clear_excluded_notes,
                          help=f"Clear exclusions: search the {count_excluded_notes()} excluded note(s) again.")
        render_actioned_filter()

        # The route's search settings, at the bottom of the Filter box.
        with st.container():
            if exact_text_only:
                all_libs_col, filter_btn_col = st.columns([1, 1.2], vertical_alignment="center")
                with all_libs_col:
                    search_all_libraries = st.checkbox(
                        "Search all libraries",
                        value=bool(load_app_settings().get("search_all_libraries", False)),
                        key="search_all_libraries",
                        on_change=persist_search_all_libraries_setting,
                        disabled=is_single_corpus_mode(),
                        help=(
                            "Looks in every visible library and shows how many matching notes each one has, so "
                            "you can switch to the right library for a semantic search. Literal matching only: "
                            "no embeddings are needed."
                        ),
                    )
                with filter_btn_col:
                    filter_only_clicked = st.button(
                        "🔎 Filter search only",
                        help=(
                            "Lists the notes that match the filter below this box, without an LLM answer. Your "
                            "question is kept but not sent. **Max results** limits the number of notes."
                        ),
                        use_container_width=True,
                    )

                rescan_all = search_all_libraries and not is_single_corpus_mode()
                rescan_libraries = scan_libraries if rescan_all else [selected_lib]
                rescan_note_col, rescan_btn_col = st.columns([1.5, 1.3], vertical_alignment="center")
                rescan_note_slot = rescan_note_col.empty()
                if rescan_btn_col.button(
                    "🔄 Rescan → Filter search",
                    use_container_width=True,
                    help=(
                        "Picks up new, edited and deleted notes in the text index — of every visible library "
                        "when **Search all libraries** is ticked, otherwise this one — then runs "
                        "**Filter search only**. Text only: embeddings wait for **⚡ Rescan** in the left panel."
                    ),
                ):
                    with st.spinner("Rescanning notes…"):
                        with get_text_rescan_scheduler().lock:
                            rescan = run_text_rescan(
                                [
                                    (
                                        library,
                                        resolve_library_registry_name(library),
                                        resolve_library_content_path(corpus_root_path, library),
                                    )
                                    for library in rescan_libraries
                                ],
                                "button",
                            )
                    for message in rescan["errors"]:
                        st.warning(f"Rescan failed for {message}")
                    filter_only_clicked = True
                rescan_note_slot.caption(
                    describe_last_text_rescan([resolve_library_registry_name(lib) for lib in rescan_libraries])
                )
                distance_map_slot = None  # literal results have no distances to show
            else:
                search_all_libraries = False  # semantic searches always use the selected library
                filter_only_clicked = False
                # Which passages count as semantic matches: a search setting, so it lives here, not in Ask.
                _cutoff_key = f"search_distance_threshold_{safe_lib}"
                if _cutoff_key not in st.session_state:
                    st.session_state[_cutoff_key] = float(distance_threshold)
                st.slider(
                    "Distance cutoff",
                    min_value=DISTANCE_SLIDER_MIN,
                    max_value=DISTANCE_SLIDER_MAX,
                    step=DISTANCE_SLIDER_STEP,
                    key=_cutoff_key,
                    on_change=sync_cutoff_from_search,
                    args=(safe_lib,),
                    help=(
                        "Semantic matches must be closer than this distance. Lower = stricter, higher = more lax. "
                        "Same setting as **Relevance cutoff** in the left panel; changes here are saved for this corpus.\n\n"
                        "After a semantic search the bar below shows where passages were found: "
                        "**green** is within the cutoff, **amber** is where to raise it to include more."
                    ),
                )
                distance_threshold = float(st.session_state[_cutoff_key])
                distance_map_slot = st.container()
                # Last in the box, under every setting it uses.
                find_clicked = st.button(
                    "🔎 Find notes",
                    key="find_notes_button",
                    use_container_width=True,
                    help="Lists the notes about what's in **Find notes about**, within **Narrow by terms** if set.",
                )
        # Scripts with no visible part: last in the Filter box, so their empty slots don't open gaps between rows.
        install_filter_commit_on_blur()
        install_tooltip_hide_on_click()
        install_save_flash()

    # Filter-only results are listed here, under the Filter box.
    filter_results_slot = st.container()

ollama_ok, installed_ollama_models = discover_ollama_models()
ollama_model_options = installed_ollama_models if ollama_ok and installed_ollama_models else SUGGESTED_OLLAMA_MODELS
default_ollama = corpus_config.get("ollama_model", DEFAULT_OLLAMA_MODEL)
if default_ollama not in ollama_model_options:
    ollama_model_options = [default_ollama] + list(ollama_model_options)

prompt_library = load_prompt_library()

with ask_box_col:
    if pick_ref is not None:
        # Pick mode: no LLM; the document being filled is shown under the Filter box instead.
        prompt_text, search_clicked, use_llm_answer, scope_clicked = "", False, False, False
        selected_ollama_model = default_ollama
        llm_max_tokens = int(corpus_config["llm_max_tokens"])
        llm_temperature = float(corpus_config["llm_temperature"])
        search_progress_slot = st.empty()
    else:
        with st.container(border=True, key="ask_box"):  # green, like the answer (route_tint_style)
            if "question_text" not in st.session_state:
                st.session_state["question_text"] = str(load_app_settings().get("last_question", ""))
            # One line: the title, Answer with LLM, then the injected prompt's links (e.g. Salesforce, the
            # connector app). The links' best guess uses the question as typed so far.
            with st.container(gap="xsmall", key="ask_top"):
                with st.container(horizontal=True, vertical_alignment="center", gap="medium", key="ask_header"):
                    st.markdown("**Ask**", width="content")
                    use_llm_answer = st.checkbox(
                        "Answer with LLM",
                        value=True,
                        key="use_llm_answer",
                        width="content",
                        help=(
                            "**Semantic search** finds relevant note chunks using embedding vectors. "
                            "**Answer with LLM** sends those chunks to Ollama to write a natural-language summary. "
                            "Search still runs first; the LLM does not replace retrieval."
                        ),
                    )
                    last_search_state = st.session_state.get(LAST_SEARCH_KEY)
                    render_prompt_links(
                        resolve_run_prompts(prompt_library, corpus_config)["tasks"],
                        prompt_library,
                        best_guess(str(st.session_state.get("question_text", "")), search_filter,
                                   last_search_state if (last_search_state or {}).get("library") == safe_lib else None),
                    )
                # The question: one line unless Bigger is on, its title and the toggle on the line above.
                with st.container(horizontal=True, vertical_alignment="center", key="question_heading"):
                    st.markdown(QUESTION_LABEL, width="stretch",
                                help="Saved locally and restored when the page is reloaded.")
                    big_question = st.toggle("Bigger", key="question_bigger", help="A taller box for long questions.")
                if not big_question:
                    st.html(QUESTION_ONE_LINE_STYLE)
                prompt_text = st.text_area(
                    QUESTION_LABEL,
                    height=QUESTION_BIG_HEIGHT if big_question else "content",
                    key="question_text",
                    on_change=persist_question,
                    label_visibility="collapsed",
                )
                remember_search_inputs()
            install_select_question_on_focus()  # no visible part: after the box, so it adds no gap above it
            render_injected_prompt_token(prompt_library)
            # Top to bottom: title, Answer with LLM and links → question → prompt library → model details → Search.

            if use_llm_answer:
                render_ask_prompt_panel(prompt_library, corpus_config, safe_lib)
                # A task prompt with its own model preselects it. The box has no key, so its default changes with
                # the prompt; picking another model here overrides it for this run.
                prompt_model = first_prompt_model(resolve_run_prompts(prompt_library, corpus_config)["tasks"])
                answer_model = prompt_model if prompt_model in ollama_model_options else default_ollama
                tokens_key = f"llm_max_tokens_{safe_lib}"
                saved_tokens = int(st.session_state.get(tokens_key, corpus_config["llm_max_tokens"]))
                with st.expander("Ollama model details", expanded=False, key="ask_model_details", icon=":material/tune:"):
                    selected_ollama_model = st.selectbox(
                        "Ollama model",
                        options=ollama_model_options,
                        index=ollama_model_options.index(answer_model),
                        help="Generative model that writes the final answer.",
                    )
                    llm_max_tokens = st.select_slider(
                        "Max tokens",
                        options=max_tokens_options(saved_tokens),
                        value=min(max(saved_tokens, TOKEN_STEPS[0]), LLM_MAX_TOKENS_LIMIT),
                        format_func=format_max_tokens,
                        key=tokens_key,
                        on_change=persist_llm_tuning,
                        args=(safe_lib,),
                        help=(
                            "Maximum length of the answer, including a reasoning model's thinking. The stops include "
                            "book lengths for scale. Long answers take a while (at 20 tokens/s, 50,000 tokens is about "
                            "40 minutes) and use more memory, because Ollama's context window grows to fit them. "
                            "Saved for this corpus."
                        ),
                    )
                    st.caption(describe_answer_length(llm_max_tokens))
                    llm_temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(corpus_config["llm_temperature"]),
                        step=0.05,
                        key=f"llm_temperature_{safe_lib}",
                        on_change=persist_llm_tuning,
                        args=(safe_lib,),
                        help=(
                            "Controls creativity. Lower values (e.g. 0.2) stay closer to your notes; higher values "
                            "are more varied. Saved for this corpus."
                        ),
                    )
                speed = last_llm_speed(selected_ollama_model)
                st.caption(
                    f"`{selected_ollama_model}` · up to {llm_max_tokens:,} tokens"
                    + (f" (≈{format_duration(llm_max_tokens / speed['tokens_per_second'])} at the last run's speed)"
                       if speed else "")
                    + f" · temperature {llm_temperature:.2f}"
                )
                if not ollama_ok:
                    st.warning(
                        f"Ollama is not reachable at `{OLLAMA_HOST}:{OLLAMA_PORT}`. "
                        "Run `./start_app.sh` or `ollama serve`, then `ollama pull llama3.2:3b`."
                    )
                elif selected_ollama_model not in installed_ollama_models:
                    st.warning(
                        f"Model `{selected_ollama_model}` is not installed locally. "
                        f"Run `ollama pull {selected_ollama_model}`."
                    )
            else:
                selected_ollama_model = default_ollama
                llm_max_tokens = int(corpus_config["llm_max_tokens"])
                llm_temperature = float(corpus_config["llm_temperature"])

            # Literal route: how many notes Ask (filtered only) would send, to compare with the in-scope notes.
            literal = (literal_matches(search_filter, safe_lib, library_folder, corpus_root_path,
                                       scan_libraries if search_all_libraries and not is_single_corpus_mode() else None)
                       if exact_text_only and has_positive_filter_terms(search_filter) else None)
            ask_limit = min(result_limit, ASK_NOTES_LIMIT)
            literal_count = (
                "" if literal is None else
                f" · {ask_limit} of {len(literal)} notes" if len(literal) > ask_limit
                else f" · {len(literal)} note{'' if len(literal) == 1 else 's'}"
            )
            search_clicked = st.button(
                f"🔍 Ask (filtered only){literal_count}" if exact_text_only else "🔍 Search",
                type="primary",
                key="search_button",
                use_container_width=True,
                help="Uses the route chosen in the Filter box: "
                + (
                    "**Literal**: answers your question from the notes matching the filter terms, sent in full."
                    if exact_text_only
                    else "**Semantic**: passages ranked by meaning."
                )
                + " Its notes are added to **In scope** under the Filter box."
                + f" Returns at most {ask_limit if exact_text_only else result_limit} results (**Max results**, in the "
                "Filter box" + (f"; Ask answers from {ASK_NOTES_LIMIT} notes at most" if exact_text_only else "") + ").",
            )
            _in_scope = gathered_notes(safe_lib)
            _kept_in_scope = kept_gathered_notes(_in_scope) if _in_scope else []
            # Only worth offering when the in-scope notes aren't just what Ask (filtered only) sends anyway.
            same_as_filtered = literal is not None and len(literal) <= ask_limit and set(_kept_in_scope) == literal
            scope_clicked = bool(_kept_in_scope) and not same_as_filtered and st.button(
                f"🗂 Ask about in-scope notes ({len(_kept_in_scope)})",
                use_container_width=True,
                help="Answers your question from every note in **In scope** (under the Filter box) that you haven't "
                "excluded, e.g. *list the actions* across two companies' conversations. Each note gets an equal "
                "share of the LLM's context.",
            )
            # Searching and answering show their progress here, right under the button that started them.
            # Between runs it shows the selected model's last speed.
            search_progress_slot = st.empty()
            if use_llm_answer:
                search_progress_slot.caption(describe_llm_speed(selected_ollama_model))

run_prompts = resolve_run_prompts(prompt_library, corpus_config)

new_search = search_clicked or filter_only_clicked or find_clicked or scope_clicked
listing = filter_only_clicked or find_clicked  # notes listed to down-select, no LLM answer
if new_search:
    # Also save on Search, in case the box was not left before clicking.
    persist_question()
    remember_searched_dates()
run_filter_only = filter_only_clicked or (search_clicked and exact_text_only)
# Filter search only lists up to Max results (up to 1000); Ask (filtered only) answers from at most ASK_NOTES_LIMIT.
notes_limit = result_limit if filter_only_clicked else min(result_limit, ASK_NOTES_LIMIT)
_gathered = gathered_notes(safe_lib)
_in_scope_before = len(_gathered["notes"]) if _gathered else 0
search_all_libraries = search_all_libraries and run_filter_only and not is_single_corpus_mode()

if new_search:
    # A new search replaces the previous results and discussion.
    st.session_state.pop(LAST_SEARCH_KEY, None)

if new_search and run_filter_only and not has_positive_filter_terms(search_filter):
    st.warning("Add at least one **Must include** or **Any of** term to filter notes.")
elif find_clicked and not find_query.strip():
    st.warning("Type what the notes are about in **Find notes about** first.")
elif scope_clicked and not prompt_text.strip() and not run_prompts["tasks"]:
    st.warning("Type a question (or choose a task) to ask about the in-scope notes.")
elif new_search and (prompt_text or run_filter_only or find_clicked or scope_clicked):
    final_results = None
    notice = ""
    nearest_distance = None
    library_counts: list[tuple[str, int]] = []
    uploads_dirs: tuple[Path, ...] = ()
    state_library = safe_lib
    try:
        if scope_clicked:
            # Every in-scope note kept, each with an equal share of the LLM's context.
            kept = kept_gathered_notes(_gathered)
            final_results, trimmed = share_context(_gathered, kept, LLM_CONTEXT_MAX_CHARS)
            if trimmed:
                notice = (f"{trimmed} of the {len(kept)} in-scope notes were trimmed so every one fits the LLM's "
                          f"context ({LLM_CONTEXT_MAX_CHARS:,} characters of notes).")
            try:
                uploads_dirs = get_library_uploads_dirs(Library().load_library(safe_lib))
            except Exception:
                uploads_dirs = ()
            if any(lib != safe_lib for lib, _source in kept):
                state_library = ALL_LIBRARIES_KEY
        elif search_all_libraries:
            state_library = ALL_LIBRARIES_KEY
            with search_progress_slot.container(), st.spinner("Searching all libraries for exact text…", show_time=True):
                final_results, library_counts = exact_text_search_all_libraries(
                    filter_libraries_for_display(all_libraries),
                    search_filter,
                    notes_limit,
                    corpus_root_path,
                    apply_exclusions=not filter_only_clicked,
                )
            if not final_results:
                notice = f"No notes in any library match {describe_search_filter(search_filter)}."
        else:
            try:
                search_lib = Library().load_library(safe_lib)
            except Exception:
                search_lib = None

            if not search_lib:
                st.error("Library not found. Use **Rescan** or **Re-index** in the left panel first.")
            elif run_filter_only:
                uploads_dirs = get_library_uploads_dirs(search_lib)
                with search_progress_slot.container(), st.spinner("Searching for exact text…", show_time=True):
                    # Filter search only lists excluded notes too, so they can be un-ticked.
                    run_filter = search_filter if filter_only_clicked else with_exclusions(search_filter, safe_lib)
                    run_filter = with_path_matches(run_filter, safe_lib, library_folder, corpus_root_path)
                    final_results = exact_text_search(safe_lib, run_filter, notes_limit)
                if not final_results:
                    notice = f"No notes match {describe_search_filter(search_filter)}."
            else:
                uploads_dirs = get_library_uploads_dirs(search_lib)
                emb_ready, emb_message = is_embedding_ready(search_lib, selected_embedding_model)
                if not emb_ready:
                    if migrate_legacy_vectors(safe_lib, selected_embedding_model):
                        emb_ready, emb_message = is_embedding_ready(search_lib, selected_embedding_model)
                if not emb_ready:
                    st.error(f"Semantic index not ready: {emb_message}")
                elif find_clicked:
                    with search_progress_slot.container(), st.spinner("Finding notes…", show_time=True):
                        # Excluded notes are listed too, so they can be un-ticked.
                        final_results, notice, nearest_distance = find_notes(
                            search_lib,
                            find_query.strip(),
                            with_path_matches(search_filter, safe_lib, library_folder, corpus_root_path),
                            selected_embedding_model,
                            result_limit,
                            distance_threshold,
                        )
                    if not final_results:
                        notice = notice or f"No notes found about *{find_query.strip()}*."
                else:
                    run_filter = with_exclusions(search_filter, safe_lib)
                    with search_progress_slot.container(), st.spinner("Searching…", show_time=True):
                        final_results, notice, nearest_distance = perform_search(
                            search_lib,
                            prompt_text,
                            with_path_matches(run_filter, safe_lib, library_folder, corpus_root_path),
                            selected_embedding_model,
                            result_limit,
                            distance_threshold,
                        )

        if final_results is not None:
            add_folder_paths(final_results, library_folder, corpus_root_path)
            route = "scope" if scope_clicked else (
                "literal" if filter_only_clicked or (search_clicked and exact_text_only) else "semantic"
            )
            if route == "literal" and not search_all_libraries:
                restrict_gathered_to_library(safe_lib)
            if final_results and not scope_clicked:  # every search adds its notes to the in-scope list
                gather_results(safe_lib, final_results, route, library_counts, uploads_dirs)
            # Corpus prompts describe one library's author role, so skip them across libraries.
            search_corpus_prompt = "" if search_all_libraries else run_prompts["corpus"]
            tasks = run_prompts["tasks"]
            agents = [task for task in tasks if is_agent_prompt(task)]
            run_references = run_reference_documents(tasks, prompt_library) if use_llm_answer else []
            # The Ask box's Salesforce chip, or a prompt that asks for Salesforce records.
            salesforce_prompts = [task["name"] for task in tasks if task.get("salesforce")]
            run_salesforce = use_llm_answer and bool(load_reference_snapshot()) and (
                bool(load_app_settings().get(SALESFORCE_CONTEXT_SETTING)) or bool(salesforce_prompts))
            search_state = {
                "filter_only": listing,
                "route": route,
                "library": state_library,
                "results": final_results,
                "question": "" if listing else prompt_text,
                "notice": notice,
                "distances": [
                    float(result["distance"])
                    for result in final_results
                    if result.get("distance") is not None
                ] or ([nearest_distance] if nearest_distance is not None else []),
                "library_counts": library_counts,
                "cutoff": distance_threshold,
                "uploads_dirs": uploads_dirs,
                "model": selected_ollama_model,
                "answer": None,
                "usage": None,
                "answer_error": None,
                "discussion": [],
                "prompt_context": build_prompt_context(
                    search_corpus_prompt,
                    run_prompts["profile"],
                    [p["text"] for p in run_prompts["globals"]],
                    [(reference_label(ref), reference_document_text(ref)[0]) for ref in run_references],
                ),
                # An Agent's first answer comes from its collection prompt (step 1), told where it's heading.
                "task_prompt": (
                    build_agent_task_prompt(agents, final_results) if agents
                    else tasks[0]["text"] if tasks else ""
                ),
                "prompt_parts": describe_run_prompts(run_prompts, search_corpus_prompt)
                + [f"Reference: {reference_label(ref)}" for ref in run_references]
                + (["Salesforce records"] if run_salesforce else []),
                "salesforce_context": run_salesforce,
                "library_display": selected_lib if state_library == safe_lib else "All libraries",
                "id": uuid.uuid4().hex[:8],  # keys this search's per-note widgets
                "left_out_notes": [],
                "prompt_model": first_prompt_model(tasks),
                # The Agent workflow (step buttons, staging, sent batches) lives under the discussion.
                "agent": (
                    {"prompts": [dict(a) for a in agents], "staged": None, "rejected": None, "needs_work": False,
                     "sent": []}
                    if use_llm_answer and agents else None
                ),
            }
            # A chosen task can drive the answer on its own, e.g. after an exact-text search.
            if final_results and use_llm_answer and (
                search_state["question"] or (tasks and not listing)
            ):
                with llm_progress(search_progress_slot, f"Generating answer with `{selected_ollama_model}`…") as report:
                    try:
                        search_state["answer"], search_state["usage"] = answer_with_ollama(
                            question=search_state["question"],
                            results=final_results,
                            model_name=selected_ollama_model,
                            max_tokens=llm_max_tokens,
                            temperature=llm_temperature,
                            corpus_prompt=with_salesforce_context(
                                search_state, search_state["prompt_context"], final_results),
                            task_prompt=search_state["task_prompt"],
                            on_progress=report,
                        )
                        for task in tasks:
                            mark_prompt_used(task["id"])
                        if run_references:
                            mark_references_used([ref["id"] for ref in run_references])
                    except Exception as llm_exc:
                        search_state["answer_error"] = str(llm_exc)
                # Keep this run's speed for the model, and show it under Search now the spinner has gone.
                record_llm_speed(selected_ollama_model, search_state["usage"])
                search_progress_slot.caption(describe_llm_speed(selected_ollama_model))
            st.session_state[LAST_SEARCH_KEY] = search_state

            if final_results and use_llm_answer and not listing:
                tuning = {"llm_max_tokens": llm_max_tokens, "llm_temperature": llm_temperature}
                # A model picked by the prompt is for that prompt; don't make it the corpus default.
                if selected_ollama_model != search_state["prompt_model"]:
                    tuning["ollama_model"] = selected_ollama_model
                save_corpus_config(safe_lib, tuning)
            # The Ask box was drawn before this search added notes: redraw so its in-scope button counts them.
            if len((gathered_notes(safe_lib) or {}).get("notes", {})) != _in_scope_before:
                st.rerun()
    except Exception as exc:
        st.error(f"Search failed: {exc}")

# Results are kept in session state so they survive reruns (follow-ups, downloads, expanders).
last_search = st.session_state.get(LAST_SEARCH_KEY)
if last_search and last_search["library"] == safe_lib and last_search.get("distances") and distance_map_slot:
    with distance_map_slot:
        render_distance_map(distance_threshold, last_search["distances"])
        target = cutoff_to_include(min(last_search["distances"]))
        if min(last_search["distances"]) >= distance_threshold and target <= DISTANCE_SLIDER_MAX:
            st.button(
                f"Set cutoff to {target:.2f}",
                key="apply_suggested_cutoff",
                on_click=set_distance_cutoff,
                args=(safe_lib, target),
                use_container_width=True,
            )
        if abs(last_search.get("cutoff", distance_threshold) - distance_threshold) > 1e-6:
            st.caption("Cutoff changed — **Search** again to apply it.")
if pick_ref is not None and st.query_params.get(DOC_PARAM):
    document_popup(pick_ref["id"], st.query_params[DOC_PARAM])
current_search = bool(last_search and last_search["library"] in (safe_lib, ALL_LIBRARIES_KEY))
answering = current_search and not last_search.get("filter_only")
gathered = gathered_notes(safe_lib)
with filter_results_slot:
    if pick_ref is not None:
        render_pick_panel(pick_ref)
    if current_search and last_search.get("filter_only") and not last_search["results"]:
        st.warning(last_search["notice"] or "No notes match the filter.")
    if gathered and not answering:
        render_gathered_notes(gathered, library_folder, safe_lib, pick_ref)
if answering:
    # Moved on to the LLM: the in-scope notes fold into one full-width bar, and the answer shows below in green.
    if gathered:
        kept = len(kept_gathered_notes(gathered))
        with st.expander(f"In scope · {len(gathered['notes'])} note(s), {kept} kept", icon=":material/filter_list:",
                         key="in_scope_folded"):
            render_gathered_notes(gathered, library_folder, safe_lib, pick_ref)
    with st.container(key=f"results_{last_search.get('route', 'semantic')}"):
        render_search_results(
            last_search,
            library_folder,
            safe_lib,
            selected_ollama_model,
            llm_max_tokens,
            llm_temperature,
            last_search.get("prompt_context", ""),
            ollama_model_options,
        )
