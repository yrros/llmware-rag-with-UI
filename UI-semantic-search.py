import re
import json
import shutil
import sys
import time
import threading
import datetime
from functools import lru_cache
import streamlit as st
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
CORPUS_SETTINGS_PATH = APP_DATA_DIR / "corpus_settings.json"
JOB_STATUS_PATH = APP_DATA_DIR / "job_status.json"
APP_SETTINGS_PATH = APP_DATA_DIR / "app_settings.json"

DEFAULT_APP_SETTINGS = {
    "use_gpu_for_embeddings": False,
    "hide_corpora_enabled": False,
    "corpus_mode": "multi",
}

CORPUS_MODE_MULTI = "multi"
CORPUS_MODE_SINGLE = "single"
SINGLE_CORPUS_DISPLAY_NAME = "All notes"
SINGLE_CORPUS_LIBRARY_NAME = "Single_Corpus"

DEFAULT_CORPUS_CONFIG = {
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "distance_threshold": DEFAULT_DISTANCE_THRESHOLD,
    "vector_db": VECTOR_DB,
    "ollama_model": DEFAULT_OLLAMA_MODEL,
    "llm_max_tokens": 512,
    "llm_temperature": 0.3,
    "hidden": False,
}

# Hardware acceleration
# - Ollama (Answer with LLM) uses its own Metal/GPU stack — independent of LLMWare config.
# - LLMWare HF embedding models only check CUDA by default; patch below can route to MPS when enabled.
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

_EMBEDDING_USE_GPU = False
_EMBEDDING_PATCH_INSTALLED = False


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
    # PyTorch MPS hits nn.Embedding bugs on Apple Silicon — keep HF models on CPU.
    return "cpu"


def resolve_embedding_device_label(use_gpu: bool) -> str:
    device = _pick_embedding_device(use_gpu)
    if use_gpu and device == "cpu":
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "cpu (MPS unsupported for HF embeddings on Mac)"
        except ImportError:
            pass
    return device


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


def format_embedding_model_option(model_id: str, model_labels: dict[str, str]) -> str:
    """Format a selectbox label with model name, size, and summary."""
    label = model_labels.get(model_id, model_id)
    summary = get_embedding_model_summary(model_id)
    return f"{label} — {summary}"


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
            help="When enabled, corpora marked hidden below are removed from the library dropdown.",
        )
        if is_hide_corpora_enabled():
            st.caption("Hidden corpora stay indexed — they are only removed from the selector.")
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


def resolve_source_path(file_source: str, library_folder: Path) -> Path | None:
    """Map a llmware file_source value to a local markdown file path."""
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

    return None


def render_source_file_link(file_source: str, library_folder: Path) -> None:
    """Render a clickable link to the source markdown file when it can be found."""
    path = resolve_source_path(file_source, library_folder)
    if path and path.exists():
        st.markdown(f"[Open note: `{path.name}`]({path.as_uri()})")
        st.caption(str(path))
    elif file_source:
        st.caption(f"Source: `{file_source}`")


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


def get_active_library_embedding(lib: Library) -> str | None:
    for emb in lib.get_embedding_status() or []:
        if emb.get("embedding_status") == "yes" and emb.get("embedding_db") == VECTOR_DB:
            return emb.get("embedding_model")
    return None


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


def hard_reset_library(name: str, embedding_model: str) -> Library:
    """Delete library artifacts, vector index, and create a fresh library."""
    safe_name = sanitize_library_name(name)

    try:
        temp_lib = Library().load_library(safe_name)
        active_model = get_active_library_embedding(temp_lib) or embedding_model
        try:
            temp_lib.delete_installed_embedding(active_model, VECTOR_DB)
        except Exception:
            pass
        temp_lib.delete_library(confirm_delete=True)
    except Exception:
        pass

    delete_vector_table(safe_name, embedding_model)
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


def get_ingest_directories(folder_path: Path) -> list[Path]:
    """Return every directory under folder_path that contains ingestible files."""
    directories = {folder_path.resolve()}
    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in INGEST_EXTENSIONS:
            directories.add(path.parent.resolve())
    return sorted(directories, key=lambda p: len(p.parts))


def ingest_all_files(lib: Library, folder_path: Path) -> dict:
    """Ingest files from all subfolders — LLMWare only scans one directory level at a time."""
    totals = {
        "docs_added": 0,
        "blocks_added": 0,
        "dirs_scanned": 0,
        "rejected_files": [],
    }

    for directory in get_ingest_directories(folder_path):
        result = lib.add_files(str(directory))
        totals["dirs_scanned"] += 1
        if not result:
            continue
        totals["docs_added"] += result.get("docs_added", 0) or 0
        totals["blocks_added"] += result.get("blocks_added", 0) or 0
        rejected = result.get("rejected_files") or []
        if isinstance(rejected, list):
            totals["rejected_files"].extend(rejected)

    return totals


def get_library_block_count(safe_lib: str) -> int:
    """Return the number of text blocks currently stored for a library."""
    try:
        lib = Library().load_library(safe_lib)
        card = lib.get_library_card()
        return int(card.get("blocks", 0)) if card else 0
    except Exception:
        return 0


def prepare_embeddings(lib: Library, embedding_model: str) -> None:
    """Clear stale embedding flags when the vector index is missing or empty."""
    if get_vector_row_count(lib.library_name, embedding_model) > 0:
        return
    try:
        lib.delete_installed_embedding(embedding_model, VECTOR_DB)
    except Exception:
        delete_vector_table(lib.library_name, embedding_model)


def sync_library(lib: Library, folder_path: Path, embedding_model: str) -> dict:
    """Ingest all files recursively, then build embeddings."""
    ingest_stats = ingest_all_files(lib, folder_path)
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
        import lancedb

        db = lancedb.connect(str(LANCEDB_PATH))
        name = get_collection_name(safe_lib, embedding_model)
        if name not in lancedb_table_names(db):
            return 0
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
    record = lib.get_embedding_status()
    if not record:
        return False, "No embedding record found."

    for emb in record:
        if emb.get("embedding_model") != embedding_model:
            continue
        if emb.get("embedding_db") != VECTOR_DB:
            continue
        if emb.get("embedding_status") != "yes":
            return False, f"Embedding status is '{emb.get('embedding_status')}'."

        expected = emb.get("embedded_blocks", 0)
        vector_rows = get_vector_row_count(lib.library_name, embedding_model)
        if vector_rows == 0:
            return False, (
                f"Vector index is empty for `{embedding_model}` "
                f"(library card shows {expected} blocks). Run **Rebuild**."
            )
        if vector_rows < expected:
            return True, f"{vector_rows}/{expected} vectors indexed ({embedding_model})"
        return True, f"{vector_rows} blocks indexed ({embedding_model})"

    active = get_active_library_embedding(lib)
    if active and active != embedding_model:
        return False, f"Index uses `{active}`. Rebuild to switch to `{embedding_model}`."

    return False, f"No `{embedding_model}` / {VECTOR_DB} embedding found."


def format_embedding_status(lib: Library, embedding_model: str) -> str:
    ready, message = is_embedding_ready(lib, embedding_model)
    return f"Ready — {message}" if ready else f"Not ready — {message}"


def get_embedding_progress(safe_lib: str, embedding_model: str) -> dict | None:
    """Read live embedding progress from LLMWare's status table."""
    status_list = Status().get_embedding_status(safe_lib, embedding_model)
    if not status_list:
        return None
    return status_list[0]


def run_with_embedding_progress(operation_label: str, safe_lib: str, embedding_model: str, work_fn) -> dict:
    """Run sync/rebuild work in a background thread and poll embedding progress every 5s."""
    error: list[Exception | None] = [None]
    ingest_stats: list[dict] = [{}]

    def worker() -> None:
        try:
            ingest_stats[0] = work_fn() or {}
        except Exception as exc:
            error[0] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    progress_bar = st.sidebar.progress(0.0, text=f"{operation_label}: starting…")
    live_status = st.sidebar.empty()

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

        chunk = f"[Source {index}: {source}]\n{text}\n"
        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts)


def answer_with_ollama(
    question: str,
    results: list,
    model_name: str,
    host: str = OLLAMA_HOST,
    port: int = OLLAMA_PORT,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> tuple[str, dict]:
    """Generate a natural-language answer from search results using a local Ollama model."""
    import requests
    from llmware.prompts import PromptCatalog

    context = build_context_from_results(results)
    if not context.strip():
        raise ValueError("No searchable text was available to send to the LLM.")

    prompt_dict = PromptCatalog().build_core_prompt(
        prompt_name="default_with_context",
        separator="\n",
        query=question,
        context=context,
        inference_dict={"temperature": temperature, "max_tokens": max_tokens},
    )
    full_prompt = prompt_dict["core_prompt"]

    url = f"http://{host}:{port}/api/chat"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
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

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {response.text[:500]}") from exc

    if data.get("error"):
        raise RuntimeError(f"Ollama error for model `{model_name}`: {data['error']}")

    message = data.get("message") or {}
    answer = str(message.get("content", "")).strip()
    if not answer:
        raise RuntimeError(
            f"Ollama model `{model_name}` returned an empty answer. "
            f"Check that the model is pulled (`ollama pull {model_name}`)."
        )

    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    output_tokens = int(data.get("eval_count") or 0)
    total_duration = data.get("total_duration")
    usage = {
        "input": prompt_tokens,
        "output": output_tokens,
        "total": prompt_tokens + output_tokens,
        "metric": "tokens",
        "processing_time": (total_duration / 1e9) if total_duration else None,
    }
    return answer, usage


def normalize_text_search_query(query: str) -> str | None:
    """Collapse whitespace and drop punctuation so LLMWare FTS gets clean tokens."""
    cleaned = re.sub(r"[^\w\s]", " ", query or "")
    tokens = [tok for tok in cleaned.split() if tok]
    return " ".join(tokens) if tokens else None


def safe_text_query(q: Query, query: str, result_count: int = 20) -> list:
    """Run a text pre-filter/fallback query, skipping empty or punctuation-only input."""
    normalized = normalize_text_search_query(query)
    if not normalized:
        return []
    results = q.text_query(normalized, result_count=result_count)
    return results or []


def perform_search(
    search_lib: Library,
    prompt_text: str,
    topic_query: str,
    embedding_model: str,
    result_limit: int,
    distance_threshold: float,
) -> tuple[list, bool]:
    """Run the hybrid search pipeline and return result blocks."""
    q = Query(
        search_lib,
        embedding_model_name=embedding_model,
        vector_db=VECTOR_DB,
    )

    filter_context = (
        safe_text_query(q, topic_query, result_count=1000)
        if normalize_text_search_query(topic_query)
        else None
    )
    filtered_ids = {res.get("doc_ID") for res in filter_context} if filter_context else None

    semantic_count = min(result_limit * 3, 100) if filtered_ids else result_limit
    broad_results = q.semantic_query(
        prompt_text,
        result_count=semantic_count,
        embedding_distance_threshold=distance_threshold,
    )

    used_text_fallback = False
    if not broad_results:
        broad_results = safe_text_query(q, prompt_text, result_count=semantic_count)
        used_text_fallback = bool(broad_results)

    if filtered_ids:
        final_results = [res for res in broad_results if res.get("doc_ID") in filtered_ids]
    else:
        final_results = broad_results

    return final_results[:result_limit], used_text_fallback


_migrated_tables = migrate_all_legacy_vectors()

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Workspace Manager", layout="wide")
st.title("🧠 RAG Workspace Manager")

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

if is_single_corpus_mode():
    selected_lib = SINGLE_CORPUS_DISPLAY_NAME
    st.sidebar.markdown(f"**Library:** {SINGLE_CORPUS_DISPLAY_NAME}")
else:
    visible_libraries = filter_libraries_for_display(all_libraries)
    if not visible_libraries:
        st.sidebar.warning("All corpora are hidden. Turn off hiding in Advanced or unhide a corpus.")
        visible_libraries = all_libraries

    selected_lib = st.sidebar.selectbox("Select Library", visible_libraries)
    if is_hide_corpora_enabled():
        hidden_count = len(all_libraries) - len(filter_libraries_for_display(all_libraries))
        if hidden_count:
            st.sidebar.caption(f"{hidden_count} corpus/corpora hidden from this list.")

safe_lib = resolve_library_registry_name(selected_lib)
library_content_path = resolve_library_content_path(corpus_root_path, selected_lib)
corpus_config = get_corpus_config(safe_lib)

model_catalog = list(load_embedding_model_catalog())
model_ids = [model["id"] for model in model_catalog]
model_labels = {model["id"]: model["label"] for model in model_catalog}

# Status display
try:
    _status_lib = Library().load_library(safe_lib)
    st.sidebar.success(f"**Status:** Loaded")
    active_model = get_active_library_embedding(_status_lib)
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
            "Global setting for all corpus scans. Uses CUDA when available. "
            "On Apple Silicon, HF embedding models stay on CPU due to a PyTorch MPS bug."
        ),
    )
    if get_use_gpu_for_embeddings() and resolve_embedding_device_label(True).startswith("cpu ("):
        st.caption("GPU toggle is on, but embedding scans will use CPU on this Mac.")
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

    distance_threshold = st.slider(
        "Relevance cutoff (distance)",
        min_value=0.30,
        max_value=1.20,
        value=float(corpus_config["distance_threshold"]),
        step=0.05,
        help="Search tuning for this corpus. Lower = stricter matches.",
        key=f"distance_threshold_{safe_lib}",
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

with st.sidebar.expander("Debug details", expanded=False):
    render_debug_panel(selected_lib, safe_lib, _status_lib, corpus_config)

st.sidebar.divider()

pending_model, pending_threshold, reembed_ack = get_pending_index_settings(safe_lib, corpus_config)
first_index = is_first_index_run(active_model)
model_change_pending = pending_model != corpus_config["embedding_model"] or bool(
    active_model and active_model != pending_model
)
sync_disabled = not first_index and model_change_pending
rebuild_disabled = not first_index and model_change_pending and not reembed_ack

if st.sidebar.button("⚡ Rescan (incremental)", disabled=sync_disabled, key=f"sync_{safe_lib}"):
    try:
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
        st.sidebar.error(f"Rescan failed: {exc}")

if st.sidebar.button("🔨 Re-index (full rebuild)", disabled=rebuild_disabled, key=f"rebuild_{safe_lib}"):
    try:
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
        st.sidebar.error(f"Re-index failed: {exc}")

render_sidebar_job_summary(selected_lib, safe_lib)

# Reload saved config for search (may have just been updated in session)
corpus_config = get_corpus_config(safe_lib)
selected_embedding_model = corpus_config["embedding_model"]
distance_threshold = float(corpus_config["distance_threshold"])
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

topic_query = st.text_input("Pre-filter keyword (optional):")
prompt_text = st.text_area("What would you like to ask?", height=100)

search_col, llm_col, limit_col = st.columns([1.2, 2.2, 1.2])
with limit_col:
    result_limit = st.number_input("Max results", min_value=1, max_value=50, value=15)
with llm_col:
    use_llm_answer = st.checkbox(
        "Answer with LLM",
        value=True,
        help=(
            "**Semantic search** finds relevant note chunks using embedding vectors. "
            "**Answer with LLM** sends those chunks to Ollama to write a natural-language summary. "
            "Search still runs first; the LLM does not replace retrieval."
        ),
    )
with search_col:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

ollama_ok, installed_ollama_models = discover_ollama_models()
ollama_model_options = installed_ollama_models if ollama_ok and installed_ollama_models else SUGGESTED_OLLAMA_MODELS
default_ollama = corpus_config.get("ollama_model", DEFAULT_OLLAMA_MODEL)
if default_ollama not in ollama_model_options:
    ollama_model_options = [default_ollama] + list(ollama_model_options)

if use_llm_answer:
    llm_cfg_col1, llm_cfg_col2, llm_cfg_col3 = st.columns([2, 1, 1])
    with llm_cfg_col1:
        selected_ollama_model = st.selectbox(
            "Ollama model",
            options=ollama_model_options,
            index=ollama_model_options.index(default_ollama),
            help="Generative model that writes the final answer.",
        )
    with llm_cfg_col2:
        llm_max_tokens = st.slider(
            "Max tokens",
            min_value=128,
            max_value=2048,
            value=int(corpus_config["llm_max_tokens"]),
            step=64,
            help="Maximum length of the generated answer. Higher values allow longer responses but take more time.",
        )
    with llm_cfg_col3:
        llm_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(corpus_config["llm_temperature"]),
            step=0.05,
            help="Controls creativity. Lower values (e.g. 0.2) stay closer to your notes; higher values are more varied.",
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

if search_clicked and prompt_text:
    try:
        search_lib = Library().load_library(safe_lib)
    except Exception:
        search_lib = None

    if not search_lib:
        st.error("Library not found. Use **Rescan** or **Re-index** in the left panel first.")
    else:
        emb_ready, emb_message = is_embedding_ready(search_lib, selected_embedding_model)
        if not emb_ready:
            if migrate_legacy_vectors(safe_lib, selected_embedding_model):
                emb_ready, emb_message = is_embedding_ready(search_lib, selected_embedding_model)
        if not emb_ready:
            st.error(f"Semantic index not ready: {emb_message}")
        else:
            with st.spinner("Searching…"):
                try:
                    final_results, used_text_fallback = perform_search(
                        search_lib,
                        prompt_text,
                        topic_query,
                        selected_embedding_model,
                        result_limit,
                        distance_threshold,
                    )

                    if not final_results:
                        st.warning("No matches found.")
                    else:
                        if used_text_fallback:
                            st.info("Vector search had no matches — showing text search results instead.")

                        if use_llm_answer:
                            with st.spinner(f"Generating answer with `{selected_ollama_model}`…"):
                                try:
                                    llm_answer, llm_usage = answer_with_ollama(
                                        question=prompt_text,
                                        results=final_results,
                                        model_name=selected_ollama_model,
                                        max_tokens=llm_max_tokens,
                                        temperature=llm_temperature,
                                    )
                                    st.subheader("Answer")
                                    st.markdown(llm_answer)

                                    cited_paths: list[Path] = []
                                    seen_paths: set[str] = set()
                                    for result in final_results:
                                        path = resolve_source_path(
                                            str(result.get("file_source", "")),
                                            library_folder,
                                        )
                                        if path and str(path) not in seen_paths:
                                            seen_paths.add(str(path))
                                            cited_paths.append(path)
                                    if cited_paths:
                                        st.caption("Sources used:")
                                        for path in cited_paths:
                                            st.markdown(f"- [`{path.name}`]({path.as_uri()})")

                                    processing_time = llm_usage.get("processing_time")
                                    if processing_time is not None:
                                        st.caption(
                                            f"Model: `{selected_ollama_model}` · "
                                            f"{llm_usage.get('total', '?')} tokens · "
                                            f"{processing_time:.1f}s"
                                        )
                                except Exception as llm_exc:
                                    st.error(f"LLM answer failed: {llm_exc}")

                        st.subheader("Sources")
                        for i, res in enumerate(final_results, 1):
                            distance = res.get("distance")
                            file_source = str(res.get("file_source", "Unknown source"))
                            label = f"#{i} — {Path(file_source).name}"
                            if distance is not None:
                                label += f" (distance: {distance:.3f})"
                            with st.expander(label):
                                render_source_file_link(file_source, library_folder)
                                st.write(res.get("text", ""))

                        if use_llm_answer:
                            save_corpus_config(
                                safe_lib,
                                {
                                    "ollama_model": selected_ollama_model,
                                    "llm_max_tokens": llm_max_tokens,
                                    "llm_temperature": llm_temperature,
                                },
                            )
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
