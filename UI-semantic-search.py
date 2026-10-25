import math
import re
import json
import gc
import shutil
import sqlite3
import sys
import time
import threading
import datetime
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


def persist_max_results() -> None:
    save_app_settings({"max_results": int(st.session_state.get("max_results", 15))})


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


# Quick date buttons under Must include. Notes carry their date as 2026-10-25, and weekly notes are headed
# "Week#43"; ISO weeks (Monday to Sunday) and ISO week-years, so 29 Dec 2025 is in 2026 Week#1.
QUICK_DATE_LABELS = {
    "today": "Today", "yesterday": "Yesterday", "this_week": "This week", "last_week": "Last week",
    "this_month": "This month", "last_month": "Last month",
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


def render_filter_heading(field: str, label: str, field_help: str) -> None:
    """A filter field's heading with its quick date buttons on the same line, and quick search terms below."""
    today = datetime.date.today()
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
        st.markdown(label, width="content", help=field_help)
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
    """Clear Filter button: empty every filter field and drop note exclusions."""
    for field in SEARCH_FILTER_FIELDS:
        st.session_state[f"filter_{field}"] = []
    save_app_settings({"search_filter": {field: [] for field in SEARCH_FILTER_FIELDS}})
    clear_excluded_notes()


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
        for prompt in library["prompts"]:
            # Libraries saved before Agent existed: prompts that already drafted output become Agents.
            prompt.setdefault("agent", prompt.get("output", "text") in DRAFT_OUTPUT_TYPES and not prompt.get("global"))
            # ...and before Agent workflows: no middle steps, a default Action Command, no model of its own.
            prompt.setdefault("steps", [])
            prompt.setdefault("command", DEFAULT_AGENT_COMMANDS.get(prompt.get("output", ""), ""))
            prompt.setdefault("model", "")
            prompt.setdefault("links", [])
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
    """Five most recently used task prompts, then the most recently used starred ones not already listed."""
    task_prompts = [p for p in library["prompts"] if not p.get("global")]
    recent = sorted(
        (p for p in task_prompts if p.get("last_used")), key=lambda p: p["last_used"], reverse=True
    )[:PROMPT_STACK_SIZE]
    recent_ids = {p["id"] for p in recent}
    starred = sorted(
        (p for p in task_prompts if p.get("starred") and p["id"] not in recent_ids),
        key=lambda p: (p.get("last_used") or "", p.get("created") or ""),
        reverse=True,
    )[:PROMPT_STACK_SIZE]
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


def build_prompt_context(corpus_prompt: str = "", profile_text: str = "", global_texts=()) -> str:
    """Preamble sent ahead of the notes: today's date, About me, corpus prompt, global prompts."""
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
        "`chase_description` (or \"\")."
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
REFERENCE_PROMPT_LIMIT = 80  # records listed to the model; the snapshot is newest-updated first
_reference_cache: dict = {}


def load_reference_snapshot() -> dict | None:
    """The connector's snapshot of open opportunities, or None if there isn't a valid one (re-read when it changes)."""
    path = get_outbox_dir() / "reference" / REFERENCE_FILE
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    if _reference_cache.get("key") != (str(path), stamp):
        try:
            snapshot = json.loads(path.read_text())
        except (OSError, ValueError):
            snapshot = None
        if not isinstance(snapshot, dict) or snapshot.get("schema") != REFERENCE_SCHEMA_ID:
            snapshot = None
        _reference_cache.update(key=(str(path), stamp), snapshot=snapshot)
    return _reference_cache["snapshot"]


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


def reference_prompt_text(records: list[dict]) -> str:
    """The open opportunities the notes mention, for the model to name in related_to (empty if there are none)."""
    records = records[:REFERENCE_PROMPT_LIMIT]
    if not records:
        return ""
    lines = "\n".join(f"- {r['name']} ({describe_reference_record(r)})" for r in records)
    return (
        "These open opportunities in the connected system are mentioned in the notes. Use this list only to "
        "name the record: when an item is about one of them, set related_to.name to its exact name and "
        "related_to.kind to \"opportunity\". Don't create an item for an opportunity just because it is listed "
        "here; only for what the notes say. If none fits, leave related_to empty rather than inventing a "
        f"name.\n{lines}"
    )


def agent_action(prompt: dict) -> str:
    return f"\"{agent_command(prompt)}\" ({OUTPUT_TYPES[get_prompt_output_type(prompt)].lower()})"


def build_agent_task_prompt(prompts: list[dict], results: list) -> str:
    """Task for an Agent's first answer: its collection prompt, plus how to list items and where it's heading.

    Several Agents (e.g. SF_Next_Step and SF_Tasks) become parts of one request, each answered under its own
    heading, with the items numbered across all parts.
    """
    reference = reference_prompt_text(notes_reference_records(results))
    tail = (
        " Ask me the questions you need answered to do that well, and don't guess anything the notes don't say."
        + (f" Say which opportunity each item belongs to.\n\n{reference}" if reference else "")
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


# --- Common links: attached to prompts, shown under the question while a prompt is injected -------------

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
# Connector app). It's defined first in Common links, listed first everywhere, and shown on the Actions page.
EXECUTOR_ROLE = "executor"


def ordered_links(library: dict) -> list[dict]:
    """Common links with the task executor first."""
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


def render_prompt_links(prompts: list[dict], library: dict, guess: str) -> None:
    """The injected prompts' links, inline (task executor first), each opening in a new tab."""
    chosen = {link_id for prompt in prompts for link_id in prompt.get("links", [])}
    if not chosen:
        return
    parts = [link_markdown(link, guess) for link in ordered_links(library) if link["id"] in chosen and link.get("url")]
    if parts:
        st.markdown(" · ".join(parts))


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
) -> dict:
    """Stage every injected Agent's records into one batch (one package). Each Agent is asked separately."""
    parts = len(prompts)
    batches = [
        stage_agent_items(search_state, prompt, model_name, max_tokens, temperature, corpus_prompt,
                          part=(number, parts) if parts > 1 else None)
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
) -> dict:
    """Turn the discussion so far into exact records for the Agent's Action Command. Nothing is sent.

    `part` is (number, of) when several Agents were injected: only this Agent's part is staged here.
    """
    item_type = get_prompt_output_type(prompt)
    messages = build_discussion_messages(search_state, max_tokens, temperature, corpus_prompt)
    reference = reference_prompt_text(notes_reference_records(discussion_results(search_state)))
    only_part = (
        f"This covers only part {part[0]} of {part[1]} ({prompt['name']}): give only that part's records; "
        "the other parts are staged separately.\n\n" if part else ""
    )
    messages.append({
        "role": "user",
        "content": (
            f"{agent_command(prompt)}\n\n{only_part}Turn what we have agreed in this conversation into records now. "
            "Include only the items I selected or agreed, with the details we agreed, and leave out anything "
            f"I said to ignore. {STRUCTURED_OUTPUT_GUIDES[item_type]} {STRUCTURED_RELATED_GUIDE}"
            + (f"\n\n{reference}" if reference else "")
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
        # v1 unless the package holds updates or mixed items, so connectors that only read v1 keep working.
        "schema": HANDOFF_SCHEMA_V2_ID if staged["type"] in ("update", "mixed") else HANDOFF_SCHEMA_ID,
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
        "items": staged["items"],
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
    "created": "✅", "updated": "✅", "exists": "✅", "needs_match": "⚠️", "invalid": "⚠️", "stale": "⚠️",
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
    label = (
        f"{icon} {format_when(entry['when'])} · {len(items)} {kind}{'' if len(items) == 1 else 's'}"
        + (f" · Agent:{prompt_name}" if prompt_name else "")
        + f" · {status_text}"
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
        for index, item in enumerate(items):
            result = results.get(index) or {}
            try:
                line = describe_staged_item(item)
            except (KeyError, TypeError):
                line = f"**{staged_item_title(item)}**"
            icon = RECEIPT_ITEM_ICONS.get(result.get("status"), "⏳" if state == "waiting" else "")
            st.markdown(f"{icon} {link_opportunity_names(line)}")
            notes_line = []
            if result.get("record_id"):
                notes_line.append(f"{result.get('object', 'Record')} `{result['record_id']}`")
            notes_line += receipt_links(result)
            if result.get("message"):
                notes_line.append(result["message"])
            candidates = ", ".join(
                f"{c.get('name', '?')} (`{c.get('id', '?')}`)" for c in result.get("candidates", []) if isinstance(c, dict)
            )
            if candidates:
                notes_line.append(f"Possible matches: {candidates}")
            if notes_line:
                st.caption(" · ".join(notes_line))
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


def render_actions_page() -> None:
    """Actions page (?page=actions): everything Agents sent to the outbox and what the connector did with it."""
    render_outbox_status(link=False)
    render_page_title("Actions")
    outbox = get_outbox_dir()
    # The task executor (defined in the prompt library's Common links) processes what's listed here.
    executor = executor_link(load_prompt_library())
    if executor and executor.get("url"):
        st.markdown(
            f"**Task executor:** {link_markdown(executor)} — processes the outbox; open it to run now or change "
            "the schedule."
        )
    else:
        st.caption("Set the **Task executor** link in the prompt library's **Common links** to open it from here.")
    st.caption(
        f"Everything Agents have sent to the outbox (`{outbox}`) and what the connector did with it. "
        "Edit a package's text before a connector acts on it, withdraw it to stop a connector acting on it, "
        "or retry one after a temporary error."
    )
    if notice := st.session_state.pop("actions_notice", None):
        st.toast(notice)
    entries = load_outbox_entries(outbox)
    # "View in outbox" in the Agent box links here with ?package=<file name>: start filtered to it.
    if (linked := st.query_params.get("package")) and "actions_query" not in st.session_state:
        st.session_state["actions_query"] = Path(linked).stem
    with st.container(horizontal=True, vertical_alignment="bottom"):
        query = st.text_input(
            "Filter",
            key="actions_query",
            placeholder="Search subjects, titles, prompts, owners and messages",
            icon=":material/search:",
        )
        st.button("Refresh", icon=":material/refresh:", key="actions_refresh")
    tabs = st.tabs([f"{OUTBOX_STATES[state][0]} ({len(entries[state])})" for state in OUTBOX_STATES])
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
    elif field in ("global", "starred", "agent"):
        value = bool(value)
    elif field == "output":
        value = value if value in DRAFT_OUTPUT_TYPES else DEFAULT_AGENT_OUTPUT
    elif field == "model":
        value = str(value or "")
    elif field == "links":
        value = [str(link_id) for link_id in (value or [])]
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
    if field in ("name", "text", "command"):
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


def render_prompt_card(prompt: dict, model_options: list[str], links: list[dict] = ()) -> None:
    prompt_id = prompt["id"]
    agent = is_agent_prompt(prompt)
    # The key gives the card a CSS class; Agent cards get a blue border (see render_prompt_library_page).
    with st.container(border=True, key=f"{'agent' if agent else 'task'}_prompt_card_{prompt_id}"):
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
                help="Starred prompts fill the Ask panel's Starred row.",
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
                "Add links in **Common links** at the top of this page.",
            )
        if agent:
            render_agent_workflow_editor(prompt)
        if prompt.get("last_used"):
            st.caption(f"Last used {prompt['last_used'].replace('T', ' ')}")


def render_prompt_library_page() -> None:
    """Prompt library page (opened in its own tab): About me, corpus prompt, task prompts."""
    library = load_prompt_library()
    add_agent_border_style("[class*='st-key-agent_prompt_card_']")
    render_outbox_status()
    render_page_title("Prompt library")
    st.caption(
        "Shape what the LLM is told when it answers from your notes. Each request is built in this order: "
        "today's date → About me → corpus prompt → global prompts → your notes and question → the task you pick."
    )

    links = library.get("links", [])
    with st.expander(f"Common links · {len(links)}", expanded=False, key="prompt_page_links_open",
                     icon=":material/link:"):
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

    outbox = get_outbox_dir()
    waiting = len(pending_outbox_files(outbox))
    with st.expander(
        f"Outbox · {waiting} waiting",
        expanded=False,
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
        expanded=False,
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

    with st.container(border=True):
        st.subheader("Task prompts")
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
            agents_only = st.toggle(
                "Agents only",
                key="prompt_page_agents_only",
                help="Show only Agent prompts, the ones that send drafts to the outbox.",
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
        shown = [
            p for p in library["prompts"]
            if prompt_matches_query(p, query) and (is_agent_prompt(p) or not agents_only)
        ]
        # "Edit prompt workflow" in the Discussion panel opens this page with ?prompt=<id>.
        focused = find_prompt(library, st.query_params.get("prompt"))
        if focused:
            shown = [focused]
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(f"Showing the workflow you opened: **{focused['name']}**.")
                st.button("Show all prompts", key="prompt_page_show_all", on_click=show_all_prompts)
        elif query.strip() or agents_only:
            st.caption(f"{len(shown)} of {len(library['prompts'])} prompts match.")
        if not shown:
            st.caption("No prompts match." if library["prompts"] else "No prompts yet — add one below.")
        model_options = prompt_model_options()
        for prompt in shown:
            render_prompt_card(prompt, model_options, ordered_links(library))

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
                st.checkbox("Starred", key="new_prompt_starred", help="Show it in the Ask panel's Starred row.")
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
            st.form_submit_button("Add prompt", type="primary", icon=":material/add:", on_click=add_prompt_from_form)
        if error := st.session_state.pop("add_prompt_error", None):
            st.warning(error)
        if added := st.session_state.pop("add_prompt_added", None):
            st.success(f"Added **{added}**. Agent prompts get their steps and Action Command on their card above.")

    install_filter_commit_on_blur()
    install_tooltip_hide_on_click()


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


def render_ask_prompt_panel(library: dict, corpus_config: dict) -> None:
    """Compact 'Prompt library' box in the Ask panel: always-included prompts, task, recent and starred."""
    profile = library["profile"]
    profile_text = str(profile.get("text", "")).strip() if profile.get("enabled") else ""
    corpus_prompt = get_active_corpus_prompt(corpus_config)
    global_prompts = [p for p in library["prompts"] if p.get("global") and p.get("text", "").strip()]
    task_prompts = [p for p in library["prompts"] if not p.get("global")]
    active = resolve_run_prompts(library, corpus_config)["tasks"]
    active_ids = {task["id"] for task in active}
    add_agent_border_style("[class*='st-key-use_agent_prompt_'] button", "[class*='st-key-active_agent_task']")
    st.html("<style>[class*='st-key-active_agent_task'] { padding: 0.1rem 0.25rem 0.1rem 0.75rem !important; }</style>")

    with st.container(border=True, gap="xsmall"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Prompt library**")
            st.link_button(
                "Manage",
                PROMPT_LIBRARY_PAGE_URL,
                icon=":material/open_in_new:",
                type="tertiary",
                help="Opens the prompt library in a new tab.",
            )

        if profile_text or corpus_prompt or global_prompts:
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.caption("Always:")
                if profile_text:
                    st.checkbox("About me", value=True, key="run_include_profile", help=profile_text)
                if corpus_prompt:
                    st.checkbox("Corpus prompt", value=True, key="run_include_corpus", help=corpus_prompt)
                for prompt in global_prompts:
                    st.checkbox(
                        prompt["name"], value=True, key=f"run_include_global_{prompt['id']}", help=prompt["text"]
                    )

        recent, starred = select_prompt_stacks(library)
        for title, prompts in (("Recent", recent), ("Starred", starred)):
            if not prompts:
                continue
            with st.container(horizontal=True, vertical_alignment="center", gap="xsmall"):
                st.caption(f"{title}:")
                for prompt in prompts:
                    is_active = prompt["id"] in active_ids
                    agent = is_agent_prompt(prompt)
                    st.button(
                        truncate_text(agent_prompt_label(prompt), 34 if agent else 28),
                        # Agent keys pick up the blue border style above.
                        key=f"use_{'agent_' if agent else ''}prompt_{title}_{prompt['id']}",
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


def render_full_document(file_source: str, path: Path | None, safe_lib: str) -> None:
    """Full note text (from disk, or rebuilt from the index) plus a download button."""
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
                key=f"download::{safe_lib}::{file_source}",
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
            corpus_prompt,
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


def render_copy_button(text: str, label: str = "Copy") -> None:
    """Small button that copies `text` to the clipboard (component iframes may write to it)."""
    render_copy_buttons([(label, text)])


def render_copy_buttons(buttons: list[tuple[str, str]]) -> None:
    """A row of small copy buttons, one per (label, text), in a single component iframe."""
    buttons = [(label, text) for label, text in buttons if text and text.strip()]
    if not buttons:
        return
    payload = json.dumps([text for _label, text in buttons]).replace("</", "<\\/")
    labels = json.dumps([label for label, _text in buttons]).replace("</", "<\\/")
    components.html(
        f"""
<div id="row" style="display:flex;gap:6px"></div>
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
) -> str:
    """Chat panel for follow-up questions about the notes from the last search; returns the follow-up model."""
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
    model_name = st.selectbox(
        "Follow-up model",
        options=options,
        key=model_key,
        on_change=persist_follow_up_model,
        args=(safe_lib,),
        help=(
            "Ollama model used for follow-up questions. It can differ from the main answer model, "
            "e.g. a larger model for deeper discussion. Saved per corpus."
        ),
    )
    in_use = len({discussion_note_key(r) for r in discussion_results(search_state)})
    total = len({discussion_note_key(r) for r in search_state["results"]})
    notes = "the notes in this search" if in_use == total else f"{in_use} of {total} notes from this search"
    caption = f"Follow-ups are answered from {notes} using `{model_name}`"
    st.caption(caption + describe_prompt_parts(search_state))
    # The task the model was given (for an Agent: its collection prompt plus how to list and where it's
    # heading), ready to paste into a flow or a test.
    render_copy_buttons([("Copy task prompt", search_state.get("task_prompt", ""))])

    view_col, popout_col = st.columns(2)
    with view_col:
        expanded = st.toggle(
            "Expand",
            key="discussion_expanded",
            help="Make the discussion panel much taller. For very long conversations, use **Pop out**.",
        )
    with popout_col:
        if st.button("⤢ Pop out", use_container_width=True, help="Open the discussion in a large window."):
            show_discussion_dialog(search_state, model_name, max_tokens, temperature, corpus_prompt)

    render_discussion_body(
        search_state,
        model_name,
        max_tokens,
        temperature,
        corpus_prompt,
        key_prefix="panel",
        # A content-sized panel can be shorter than the default box, so expanding uses a fixed taller height.
        height=DISCUSSION_EXPANDED_HEIGHT if expanded else DISCUSSION_HEIGHT,
    )
    return model_name


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
) -> None:
    """Transcript, follow-up box and clear button, shared by the side panel and the pop-out."""
    discussion = search_state["discussion"]
    with st.container(height=height, border=True, autoscroll=True):
        if not discussion:
            st.caption("No follow-up questions yet.")
        for turn in discussion:
            with st.chat_message(turn["role"]):
                st.markdown(link_opportunity_names(turn["content"]))
                if turn.get("model"):
                    st.caption(f"`{turn['model']}`")
                # Every turn can be copied. When the model was sent more than is shown (a step's prompt, or
                # Needs work with the rejected list), that full prompt can be copied too, to reuse in flows.
                buttons = [("Copy", turn["content"])]
                if turn.get("prompt") and turn["prompt"] != turn["content"]:
                    buttons.append(("Copy prompt sent", turn["prompt"]))
                render_copy_buttons(buttons)

    with st.container(border=True):
        # The toggle sits outside the form: a widget inside a form only takes effect when the form is sent.
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("Ask a follow-up", width="stretch")
            big_box = st.toggle(
                "Expand",
                key=f"{key_prefix}_follow_up_expanded",
                help="Make the follow-up box bigger for longer replies.",
            )
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
        st.rerun(scope=rerun_scope)

    if submitted and follow_up.strip():
        if add_discussion_turn(search_state, follow_up.strip(), model_name, max_tokens, temperature, corpus_prompt):
            st.rerun(scope=rerun_scope)


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


def render_library_counts(search_state: dict) -> None:
    """Per-library match counts for all-library searches, with buttons to switch library."""
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
                    key=f"use_library_{library}",
                    on_click=switch_to_library,
                    args=(library,),
                    disabled=library == st.session_state.get("selected_library"),
                    use_container_width=True,
                    help="Switch to this library, e.g. to run a semantic search inside it.",
                )


def render_filter_results(search_state: dict, library_folder: Path, safe_lib: str) -> None:
    """Compact list of notes from a Filter search only run, one expandable row per note."""
    results = search_state["results"]
    if not results:
        st.warning(search_state["notice"] or "No notes match the filter.")
        return

    render_library_counts(search_state)

    notes: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        notes.setdefault((result.get("_library", ""), str(result.get("file_source", ""))), []).append(result)
    st.markdown(f"**{len(notes)} matching note(s)**")
    st.caption(
        "Your question was not sent to the LLM. Tick **🚫** to exclude a note, or click **🎯** to keep "
        "just that note (and any others you 🎯), then click **Search** to run the full workflow on the rest."
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

    # Grouped under their folder path, sorted by path; notes whose folder isn't known come last.
    folders: dict[str, list] = {}
    for note_key, blocks in notes.items():
        folders.setdefault(blocks[0].get("_folder_path", ""), []).append((note_key, blocks))
    for folder in sorted(folders, key=lambda path: (not path, path.lower())):
        st.markdown(f"📁 **{folder or 'Folder not known'}** · {len(folders[folder])}")
        for (library, file_source), blocks in folders[folder]:
            render_filter_result_row(library, file_source, blocks, listed, library_folder, safe_lib, search_state)


def render_filter_result_row(
    library: str, file_source: str, blocks: list, listed: list, library_folder: Path, safe_lib: str,
    search_state: dict,
) -> None:
    """One note in the Filter search list: exclude box, target button and an expander with the note."""
    note_lib = blocks[0].get("_safe_lib", safe_lib)
    excluded = file_source in get_excluded_notes(note_lib)
    name = Path(file_source).name
    label = (f"~~{name}~~ · excluded" if excluded else name) + (
        f"  ·  {library}" if library and not blocks[0].get("_folder_path") else ""
    )
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
            render_full_document(file_source, path, note_lib)


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


def unstage_item(agent: dict, index: int) -> None:
    agent["staged"]["items"].pop(index)
    # Item positions shift, so drop the record pickers' remembered choices for this staging.
    prefix = f"agent_ref_{agent['staged']['id']}_"
    for key in [k for k in st.session_state if str(k).startswith(prefix)]:
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
        candidates = ", ".join(c["name"] for c in result.get("candidates", []))
        st.caption(f"{title}: {result['message']}" + (f" Possible matches: {candidates}." if candidates else ""))
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
        st.caption(
            "Reply in the discussion to pick items (e.g. *Yes:1,2,3 ignore 4,5,6*), set dates and answer questions. "
            f"When it's right, stage the {item_kind}. Untick notes under Sources to narrow what the LLM uses."
        )
        snapshot = load_reference_snapshot()
        if snapshot:
            refreshed = datetime.datetime.fromisoformat(snapshot["generated_at"]).strftime("%d %b %H:%M")
            st.caption(
                f"Reference: {len(reference_records())} {snapshot.get('scope', 'records')} from "
                f"{snapshot.get('source', 'the connector')} ({snapshot.get('target', '?')}), refreshed {refreshed}. "
                "The Agent uses their exact names; refresh it from the task executor."
            )
        else:
            st.caption("No reference snapshot yet: the task executor can refresh one so the Agent uses real record names.")
        if st.button(
            label,
            key="agent_run",
            icon=":material/smart_toy:",
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
                        search_state, prompts, model_name, max_tokens, temperature, corpus_prompt
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
                        st.markdown(link_opportunity_names(describe_staged_item(item)))
                        render_reference_picker(agent, index, item)
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
        if search_state["answer"]:
            st.subheader("Answer")
            st.markdown(link_opportunity_names(search_state["answer"]))
            render_copy_buttons([
                ("Copy answer", search_state["answer"]),
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
                render_source_document(
                    file_source,
                    hits,
                    hit_folder,
                    hit_lib,
                    hit_uploads,
                )

    with chat_col:
        follow_up_model = render_follow_up_discussion(
            search_state,
            model_name,
            max_tokens,
            temperature,
            corpus_prompt,
            model_options,
            safe_lib,
        )
        if search_state.get("agent"):
            render_agent_panel(search_state, notes_in_use, follow_up_model, max_tokens, temperature, corpus_prompt)


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
) -> tuple[str, dict]:
    """Send a chat conversation to a local Ollama model and return (reply, usage).

    `response_format` is a JSON schema; Ollama then constrains the reply to match it.
    The reply is streamed, so long answers have no overall time limit; `on_progress(tokens, seconds)`
    is called every LLM_PROGRESS_SECONDS with the tokens generated so far and the seconds since the
    first one (None while the model is still reading the prompt).
    """
    import requests

    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": context_window_for(messages, max_tokens, num_ctx),
    }

    url = f"http://{host}:{port}/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    if response_format:
        payload["format"] = response_format

    try:
        # (connect, longest wait between streamed pieces): a big prompt can take minutes before the first token.
        response = requests.post(url, json=payload, stream=True, timeout=(10, 900))
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama at `{host}:{port}`. "
            f"Start Ollama or run `./start_app.sh`. Details: {exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code} for model `{model_name}`: "
            f"{response.text[:500]}"
        )

    content, thinking, data = [], [], {}
    generated, first_token_at = 0, None
    last_report = time.monotonic()
    try:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError as exc:
                raise RuntimeError(f"Ollama returned invalid JSON: {line[:500]!r}") from exc
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
    except requests.RequestException as exc:
        raise RuntimeError(f"Lost the connection to Ollama while `{model_name}` was answering: {exc}") from exc

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
    return {**search_filter, "exclude_sources": get_excluded_notes(lib_key)}


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
    picks = st.session_state.setdefault(TARGET_PICKS_KEY, set())
    picks.add((lib_key, file_source))
    for lib, source in notes:
        _set_note_excluded(lib, source, (lib, source) not in picks)
    persist_excluded_notes()


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
    having, params = build_note_filter_sql(search_filter)
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
        return [], f"No notes match the filter {describe_search_filter(search_filter)}."

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
    q = Query(
        search_lib,
        embedding_model_name=embedding_model,
        vector_db=VECTOR_DB,
    )

    if is_filter_active(search_filter):
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
st.set_page_config(page_title="RAG Workspace Manager", layout="wide")
# Starts the background auto-rescan thread once per server process.
get_text_rescan_scheduler()
if st.query_params.get("page") == "prompts":
    render_prompt_library_page()
    st.stop()
if st.query_params.get("page") == "actions":
    render_actions_page()
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

if st.sidebar.button("⚡ Rescan (incremental)", disabled=sync_disabled, key=f"sync_{safe_lib}"):
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

if not is_single_corpus_mode():
    st.sidebar.markdown("**All corpora**")
    if st.sidebar.button(
        "⚡ Rescan all corpora",
        key="multi_rescan_start",
        disabled=not scan_libraries,
        help=(
            f"Rescans the {len(scan_libraries)} visible sub-corpora (hidden ones are skipped) in turn, like the "
            "Getting started wizard: new and edited notes are indexed and embedded, deleted notes "
            "dropped. Each corpus keeps its own embedding engine and nothing is wiped — only changes "
            "are processed, so after the first run this is quick."
        ),
    ):
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

filter_box_col, ask_box_col = st.columns(2, gap="medium")

with filter_box_col:
    with st.container(border=True):
        title_col, clear_filter_col = st.columns([3, 1.2], vertical_alignment="center")
        title_col.markdown("**Filter**")
        clear_filter_col.button(
            "Clear Filter",
            on_click=clear_search_filter,
            use_container_width=True,
            help="Remove all filter terms and note exclusions.",
        )
        # Two routes into the notes: semantic (vector) search or literal text matching. The controls below
        # follow the route; the Ask box's Search button uses it too.
        if "search_route" not in st.session_state:
            st.session_state["search_route"] = saved_search_route()
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
        for field, (label, field_help) in FILTER_WIDGETS.items():
            _key = f"filter_{field}"
            if _key not in st.session_state:
                st.session_state[_key] = clean_filter_terms(_saved_filter.get(field, []))
                # Dates restored from the last session count as left over from an earlier search.
                st.session_state[f"searched_dates_{field}"] = date_terms(st.session_state[_key])
            # The heading carries the field's help and its quick date buttons, on one line.
            render_filter_heading(field, label, (
                f"{field_help}\n\n"
                "Each term is matched **literally** (case-insensitive) anywhere in the note or its "
                "file name, so `2026-10` matches `2026-10-19` but not `2026 … 10`. "
                + (
                    "On the **Literal** route, **Search** returns the matching notes in full."
                    if exact_text_only
                    else "On the **Semantic** route, passages are ranked **only within** the matching notes."
                )
            ))
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
        if "max_results" not in st.session_state:  # saved between searches and sessions
            st.session_state["max_results"] = min(max(int(load_app_settings().get("max_results", 15) or 15), 1), 50)
        result_limit = st.number_input(
            "Max results",
            min_value=1,
            max_value=50,
            key="max_results",
            on_change=persist_max_results,
            width=160,
            help="The most passages (Semantic route) or notes (Literal route) a search returns. Saved for next time.",
        )
        install_filter_commit_on_blur()
        install_tooltip_hide_on_click()
        search_filter = get_search_filter_from_widgets()
        if is_filter_active(search_filter):
            st.caption(f"Filter: {describe_search_filter(search_filter)}")
        if count_excluded_notes():
            excluded_col, clear_col = st.columns([3, 1.4], vertical_alignment="center")
            excluded_col.caption(f"🚫 {count_excluded_notes()} note(s) excluded from **Search**.")
            clear_col.button("Clear exclusions", on_click=clear_excluded_notes, use_container_width=True)

        with st.container(border=True):
            st.markdown(
                "**Advanced search settings** · "
                + ("literal: whole notes that match the filter" if exact_text_only else "semantic: passages ranked by meaning")
            )
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
                rescan_note_col, rescan_btn_col = st.columns([2, 1.2], vertical_alignment="center")
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

    # Filter-only results are listed here, under the Filter box.
    filter_results_slot = st.container()

ollama_ok, installed_ollama_models = discover_ollama_models()
ollama_model_options = installed_ollama_models if ollama_ok and installed_ollama_models else SUGGESTED_OLLAMA_MODELS
default_ollama = corpus_config.get("ollama_model", DEFAULT_OLLAMA_MODEL)
if default_ollama not in ollama_model_options:
    ollama_model_options = [default_ollama] + list(ollama_model_options)

prompt_library = load_prompt_library()

with ask_box_col:
    with st.container(border=True):
        st.markdown("**Ask**")
        if "question_text" not in st.session_state:
            st.session_state["question_text"] = str(load_app_settings().get("last_question", ""))
        install_select_question_on_focus()
        prompt_text = st.text_area(
            QUESTION_LABEL,
            height=100,
            key="question_text",
            on_change=persist_question,
            help="Saved locally and restored when the page is reloaded.",
        )
        # The injected prompt's links (e.g. Salesforce, the connector app), between the question and its token.
        last_search_state = st.session_state.get(LAST_SEARCH_KEY)
        render_prompt_links(
            resolve_run_prompts(prompt_library, corpus_config)["tasks"],
            prompt_library,
            best_guess(prompt_text, search_filter, last_search_state if (last_search_state or {}).get("library") == safe_lib else None),
        )
        render_injected_prompt_token(prompt_library)
        # Top to bottom: question → Answer with LLM → prompt library → model details → cutoff → Search → progress.
        use_llm_answer = st.checkbox(
            "Answer with LLM",
            value=True,
            key="use_llm_answer",
            help=(
                "**Semantic search** finds relevant note chunks using embedding vectors. "
                "**Answer with LLM** sends those chunks to Ollama to write a natural-language summary. "
                "Search still runs first; the LLM does not replace retrieval."
            ),
        )

        if use_llm_answer:
            render_ask_prompt_panel(prompt_library, corpus_config)
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

        search_clicked = st.button(
            "🔍 Ask (filtered only)" if exact_text_only else "🔍 Search",
            type="primary",
            use_container_width=True,
            help="Uses the route chosen in the Filter box: "
            + (
                "**Literal**: answers your question from the notes matching the filter terms, sent in full."
                if exact_text_only
                else "**Semantic**, by meaning."
            )
            + f" Returns at most {result_limit} results (**Max results**, in the Filter box).",
        )
        # Searching and answering show their progress here, right under the button that started them.
        # Between runs it shows the selected model's last speed.
        search_progress_slot = st.empty()
        if use_llm_answer:
            search_progress_slot.caption(describe_llm_speed(selected_ollama_model))

run_prompts = resolve_run_prompts(prompt_library, corpus_config)

new_search = search_clicked or filter_only_clicked
if new_search:
    # Also save on Search, in case the box was not left before clicking.
    persist_question()
    remember_searched_dates()
run_filter_only = filter_only_clicked or (search_clicked and exact_text_only)
search_all_libraries = search_all_libraries and run_filter_only and not is_single_corpus_mode()

if new_search:
    # A new search replaces the previous results and discussion.
    st.session_state.pop(LAST_SEARCH_KEY, None)

if new_search and run_filter_only and not has_positive_filter_terms(search_filter):
    st.warning("Add at least one **Must include** or **Any of** term to filter notes.")
elif new_search and (prompt_text or run_filter_only):
    final_results = None
    notice = ""
    nearest_distance = None
    library_counts: list[tuple[str, int]] = []
    uploads_dirs: tuple[Path, ...] = ()
    state_library = safe_lib
    try:
        if search_all_libraries:
            state_library = ALL_LIBRARIES_KEY
            with search_progress_slot.container(), st.spinner("Searching all libraries for exact text…", show_time=True):
                final_results, library_counts = exact_text_search_all_libraries(
                    filter_libraries_for_display(all_libraries),
                    search_filter,
                    result_limit,
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
                    final_results = exact_text_search(safe_lib, run_filter, result_limit)
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
                else:
                    with search_progress_slot.container(), st.spinner("Searching…", show_time=True):
                        final_results, notice, nearest_distance = perform_search(
                            search_lib,
                            prompt_text,
                            with_path_matches(
                                with_exclusions(search_filter, safe_lib), safe_lib, library_folder, corpus_root_path
                            ),
                            selected_embedding_model,
                            result_limit,
                            distance_threshold,
                        )

        if final_results is not None:
            add_folder_paths(final_results, library_folder, corpus_root_path)
            # Corpus prompts describe one library's author role, so skip them across libraries.
            search_corpus_prompt = "" if search_all_libraries else run_prompts["corpus"]
            tasks = run_prompts["tasks"]
            agents = [task for task in tasks if is_agent_prompt(task)]
            search_state = {
                "filter_only": filter_only_clicked,
                "library": state_library,
                "results": final_results,
                "question": "" if filter_only_clicked else prompt_text,
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
                ),
                # An Agent's first answer comes from its collection prompt (step 1), told where it's heading.
                "task_prompt": (
                    build_agent_task_prompt(agents, final_results) if agents
                    else tasks[0]["text"] if tasks else ""
                ),
                "prompt_parts": describe_run_prompts(run_prompts, search_corpus_prompt),
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
                search_state["question"] or (tasks and not filter_only_clicked)
            ):
                with llm_progress(search_progress_slot, f"Generating answer with `{selected_ollama_model}`…") as report:
                    try:
                        search_state["answer"], search_state["usage"] = answer_with_ollama(
                            question=search_state["question"],
                            results=final_results,
                            model_name=selected_ollama_model,
                            max_tokens=llm_max_tokens,
                            temperature=llm_temperature,
                            corpus_prompt=search_state["prompt_context"],
                            task_prompt=search_state["task_prompt"],
                            on_progress=report,
                        )
                        for task in tasks:
                            mark_prompt_used(task["id"])
                    except Exception as llm_exc:
                        search_state["answer_error"] = str(llm_exc)
                # Keep this run's speed for the model, and show it under Search now the spinner has gone.
                record_llm_speed(selected_ollama_model, search_state["usage"])
                search_progress_slot.caption(describe_llm_speed(selected_ollama_model))
            st.session_state[LAST_SEARCH_KEY] = search_state

            if final_results and use_llm_answer and not filter_only_clicked:
                tuning = {"llm_max_tokens": llm_max_tokens, "llm_temperature": llm_temperature}
                # A model picked by the prompt is for that prompt; don't make it the corpus default.
                if selected_ollama_model != search_state["prompt_model"]:
                    tuning["ollama_model"] = selected_ollama_model
                save_corpus_config(safe_lib, tuning)
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
if last_search and last_search["library"] in (safe_lib, ALL_LIBRARIES_KEY) and last_search.get("filter_only"):
    with filter_results_slot:
        render_filter_results(last_search, library_folder, safe_lib)
elif last_search and last_search["library"] in (safe_lib, ALL_LIBRARIES_KEY):
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
