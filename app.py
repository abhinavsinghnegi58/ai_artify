"""AI-Artify RAG-enabled Flask backend with FAISS retrieval, judgments, and evaluation."""

from __future__ import annotations

import base64
import datetime
import hashlib
import logging
import math
import os
import pickle
import secrets
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import click
import faiss
import numpy as np
import requests
from filelock import FileLock
from flask import (
    Flask,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask.cli import with_appcontext
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf
from openai import BadRequestError, OpenAI
from sqlalchemy import String, inspect, text
from werkzeug.security import check_password_hash, generate_password_hash

from utils.image_enhancer import enhance_image
from utils.prompt_optimizer import optimize_prompt, optimize_with_context

# ----------------------------
# Constants / Configuration
# ----------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DEFAULT_RETRIEVAL_K = 5
FAISS_INDEX_FILENAME = "faiss.index"
FAISS_MAPPING_FILENAME = "faiss_mapping.pkl"
FAISS_LOCK_FILENAME = "faiss.lock"
IMAGE_MODEL = "gpt-image-1"
DOCUMENT_LIMIT_PER_USER = 200
DOCS_PAGE_SIZE = 10
GALLERY_PAGE_SIZE = 12
GENERATION_SESSION_KEY = "ai_artify_generation_result"

db = SQLAlchemy()
_schema_bootstrapped = False
csrf = CSRFProtect()

app = Flask(__name__, instance_relative_config=True)
os.makedirs(app.instance_path, exist_ok=True)

app.config["SECRET_KEY"] = (
    os.environ.get("FLASK_SECRET_KEY")
    or os.environ.get("SECRET_KEY")
    or secrets.token_urlsafe(32)
)
if "FLASK_SECRET_KEY" not in os.environ and "SECRET_KEY" not in os.environ:
    logging.warning(
        "FLASK_SECRET_KEY not set; using an ephemeral key. Sessions will reset on restart."
    )

database_path = Path(app.instance_path) / "database.db"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{database_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

legacy_database_path = Path(app.root_path) / "database.db"
if legacy_database_path.exists() and not database_path.exists():
    shutil.move(str(legacy_database_path), str(database_path))
    logging.info("Migrated legacy database.db into the instance folder.")

upload_folder = Path(app.instance_path) / "uploads" / "images"
upload_folder.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(upload_folder)

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    logging.warning("OPENAI_API_KEY not set; OpenAI-dependent features are disabled.")
app.config["OPENAI_API_KEY"] = openai_api_key
app.config["OPENAI_CLIENT"] = OpenAI(api_key=openai_api_key) if openai_api_key else None

db.init_app(app)
csrf.init_app(app)


@app.context_processor
def inject_csrf_token() -> Dict[str, Any]:
    return {"csrf_token": generate_csrf}

# ----------------------------
# Database Models
# ----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    images = db.relationship("Image", backref="user", lazy=True)
    documents = db.relationship("Document", backref="user", lazy=True)
    judgments = db.relationship("Judgment", backref="user", lazy=True)


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    text_hash = db.Column(db.String(64), nullable=True, index=True)
    is_public = db.Column(db.Boolean, default=False, nullable=False, index=True)


class Judgment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    query_text = db.Column("query", db.Text, nullable=False)
    doc_id = db.Column(db.Integer, db.ForeignKey("document.id"), nullable=False)
    binary = db.Column(db.Boolean, nullable=True)
    grade = db.Column(db.Integer, nullable=True)
    rank = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)

    document = db.relationship("Document", backref="judgments")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ----------------------------
# Schema Management
# ----------------------------
def ensure_database_schema(force: bool = False) -> None:
    """Create or upgrade database tables as needed."""
    engine = db.engine
    inspector = inspect(engine)

    existing_tables = set(inspector.get_table_names())
    required_tables = {"user", "image", "document", "judgment"}
    if force or required_tables - existing_tables:
        db.create_all()
        inspector = inspect(engine)

    table_names = inspector.get_table_names()
    if "image" in table_names and _image_prompt_column_needs_upgrade(inspector):
        _upgrade_image_prompt_column(engine)
    if "document" in table_names:
        if _document_text_hash_needs_upgrade(inspector):
            _upgrade_document_text_hash(engine)
        if _document_public_flag_needs_upgrade(inspector):
            _upgrade_document_public_flag(engine)


def _image_prompt_column_needs_upgrade(inspector) -> bool:
    try:
        columns = inspector.get_columns("image")
    except Exception:  # pragma: no cover - defensive fallback
        return False

    for column in columns:
        if column["name"] != "prompt":
            continue
        col_type = column["type"]
        length = getattr(col_type, "length", None)
        if isinstance(col_type, String) and length and length < 1024:
            return True
    return False


def _upgrade_image_prompt_column(engine) -> None:
    if engine.url.drivername != "sqlite":
        raise RuntimeError("Prompt column auto-migration currently supports SQLite only.")

    current_app.logger.info("Upgrading image.prompt column to TEXT")
    with engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE image RENAME TO image_old;")
        Image.__table__.create(bind=connection, checkfirst=False)
        connection.exec_driver_sql(
            """
            INSERT INTO image (id, filename, prompt, timestamp, user_id)
            SELECT id, filename, prompt, timestamp, user_id FROM image_old;
            """
        )
        connection.exec_driver_sql("DROP TABLE image_old;")
    current_app.logger.info("image.prompt column upgrade completed")


def _document_text_hash_needs_upgrade(inspector) -> bool:
    try:
        columns = inspector.get_columns("document")
    except Exception:  # pragma: no cover
        return False
    return not any(column["name"] == "text_hash" for column in columns)


def _upgrade_document_text_hash(engine) -> None:
    if engine.url.drivername != "sqlite":
        raise RuntimeError("Document hash auto-migration currently supports SQLite only.")

    with engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE document ADD COLUMN text_hash VARCHAR(64)")

    needs_backfill = (
        Document.query.filter(
            (Document.text_hash.is_(None)) | (Document.text_hash == "")
        ).all()
    )
    for doc in needs_backfill:
        doc.text_hash = _hash_text(doc.text)
    if needs_backfill:
        db.session.commit()


def _document_public_flag_needs_upgrade(inspector) -> bool:
    try:
        columns = inspector.get_columns("document")
    except Exception:  # pragma: no cover
        return False
    return not any(column["name"] == "is_public" for column in columns)


def _upgrade_document_public_flag(engine) -> None:
    if engine.url.drivername != "sqlite":
        raise RuntimeError("Document public flag auto-migration currently supports SQLite only.")

    with engine.begin() as connection:
        connection.exec_driver_sql(
            "ALTER TABLE document ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT 0"
        )


def bootstrap_schema_if_needed() -> None:
    global _schema_bootstrapped
    if _schema_bootstrapped:
        return
    ensure_database_schema()
    _schema_bootstrapped = True


def _openai_available() -> bool:
    return current_app.config.get("OPENAI_CLIENT") is not None


# ----------------------------
# FAISS + Embeddings Helpers
# ----------------------------
def _faiss_paths() -> Tuple[Path, Path]:
    index_path = Path(app.instance_path) / FAISS_INDEX_FILENAME
    mapping_path = Path(app.instance_path) / FAISS_MAPPING_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return index_path, mapping_path


def _faiss_lock() -> FileLock:
    lock_path = Path(app.instance_path) / FAISS_LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_path), timeout=10)


def load_index() -> Tuple[faiss.Index, List[int]]:
    """Load (or initialize) the FAISS index and id mapping."""
    with _faiss_lock():
        return _read_index_files()


def save_index(index: faiss.Index, mapping: List[int]) -> None:
    """Persist the FAISS index and id mapping to disk."""
    with _faiss_lock():
        _write_index_files(index, mapping)


def embed_text(text: str) -> np.ndarray:
    """Embed and normalize text using the configured OpenAI client."""
    client: OpenAI | None = current_app.config.get("OPENAI_CLIENT")
    if client is None:
        raise RuntimeError("OpenAI API key missing; embeddings are disabled.")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vector = np.array(response.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def add_document_to_index(doc_id: int, text: str) -> None:
    """Embed a document and add it to the FAISS index."""
    if not text.strip():
        return
    vector = embed_text(text)
    with _faiss_lock():
        index, mapping = _read_index_files()
        index.add(np.array([vector], dtype="float32"))
        mapping.append(doc_id)
        _write_index_files(index, mapping)


def search_index(query_text: str, top_k: int = DEFAULT_RETRIEVAL_K) -> List[Tuple[int, float]]:
    """Search the FAISS index and return doc_id-score tuples."""
    vector = embed_text(query_text)
    with _faiss_lock():
        index, mapping = _read_index_files()
    if index.ntotal == 0:
        return []
    top_k = min(max(top_k, 1), index.ntotal)
    scores, indices = index.search(np.array([vector], dtype="float32"), top_k)
    matches: List[Tuple[int, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        if idx >= len(mapping):
            continue
        matches.append((mapping[idx], float(score)))
    return matches


def retrieve_documents_for_user(user_id: int, query_text: str, k: int) -> List[Dict[str, Any]]:
    """Retrieve top-k documents for a given user, filtering out others."""
    if not query_text.strip():
        return []
    raw_matches = search_index(query_text, top_k=max(k * 3, k))
    docs: List[Dict[str, Any]] = []
    for doc_id, score in raw_matches:
        doc = Document.query.get(doc_id)
        if not doc or (doc.user_id != user_id and not doc.is_public):
            continue
        docs.append(
            {
                "id": doc.id,
                "text": doc.text,
                "score": round(score, 4),
                "is_public": doc.is_public,
                "owner_name": doc.user.username if doc.user else "Unknown",
                "owned_by_current_user": doc.user_id == user_id,
            }
        )
        if len(docs) >= k:
            break
    for idx, item in enumerate(docs, start=1):
        item["rank"] = idx
    return docs


def _serialize_retrieval(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for doc in docs:
        payload.append(
            {
                "doc_id": doc["id"],
                "rank": doc.get("rank"),
                "score": doc.get("score"),
                "is_public": doc.get("is_public", False),
                "owner_name": doc.get("owner_name"),
            }
        )
    return payload


def _hydrate_retrieval(payload: List[Dict[str, Any]], user_id: int) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for item in payload:
        doc = Document.query.get(item["doc_id"])
        if not doc or (doc.user_id != user_id and not doc.is_public):
            continue
        hydrated.append(
            {
                "id": doc.id,
                "text": doc.text,
                "rank": item.get("rank"),
                "score": item.get("score"),
                "is_public": doc.is_public,
                "owner_name": doc.user.username if doc.user else "Unknown",
                "owned_by_current_user": doc.user_id == user_id,
            }
        )
    return hydrated


def _read_index_files() -> Tuple[faiss.Index, List[int]]:
    index_path, mapping_path = _faiss_paths()
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
        except Exception:
            current_app.logger.exception("Failed to load FAISS index. Rebuilding.")
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    if mapping_path.exists():
        try:
            with mapping_path.open("rb") as fh:
                mapping = pickle.load(fh)
        except Exception:
            current_app.logger.exception("Failed to load FAISS mapping. Resetting.")
            mapping = []
    else:
        mapping = []

    if index.ntotal != len(mapping):
        current_app.logger.warning(
            "FAISS mapping length mismatch detected; resetting index and mapping."
        )
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        mapping = []
        _write_index_files(index, mapping)
    return index, mapping


def _write_index_files(index: faiss.Index, mapping: List[int]) -> None:
    index_path, mapping_path = _faiss_paths()
    faiss.write_index(index, str(index_path))
    with mapping_path.open("wb") as fh:
        pickle.dump(mapping, fh)


# ----------------------------
# CLI Command + Hooks
# ----------------------------
@click.command("init-db")
@with_appcontext
def init_db_command() -> None:
    """Initialize or upgrade the application database."""
    ensure_database_schema(force=True)
    click.echo("Database initialized.")


app.cli.add_command(init_db_command)


@app.before_request
def _ensure_schema() -> None:
    bootstrap_schema_if_needed()


# ----------------------------
# Auth Routes
# ----------------------------
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("index"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed = generate_password_hash(password)
        if User.query.filter_by(username=username).first():
            return render_template("register.html", error="Username already exists.")
        new_user = User(username=username, password=hashed)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ----------------------------
# Document Management + Retrieval
# ----------------------------
@app.route("/docs", methods=["GET", "POST"])
def docs():
    if "user_id" not in session:
        return redirect(url_for("login"))

    doc_error = None
    user_id = session["user_id"]

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        share_public = bool(request.form.get("share_public"))
        if not text:
            doc_error = "Please provide some text before saving."
        else:
            text_hash = _hash_text(text)
            doc_count = Document.query.filter_by(user_id=user_id).count()
            if doc_count >= DOCUMENT_LIMIT_PER_USER:
                doc_error = (
                    f"Document limit of {DOCUMENT_LIMIT_PER_USER} reached. "
                    "Delete older items before adding new ones."
                )
            else:
                existing = Document.query.filter_by(
                    user_id=user_id, text_hash=text_hash
                ).first()
                if existing:
                    doc_error = "This document already exists in your knowledge base."
                else:
                    try:
                        doc = Document(
                            text=text,
                            user_id=user_id,
                            text_hash=text_hash,
                            is_public=share_public,
                        )
                        db.session.add(doc)
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
                        current_app.logger.exception("Unable to save document.")
                        doc_error = "Unable to save document. Please retry."
                    else:
                        if _openai_available():
                            try:
                                add_document_to_index(doc.id, doc.text)
                            except Exception:
                                current_app.logger.exception("Failed to index document text.")
                                flash(
                                    "Document saved but embedding or indexing failed.",
                                    "error",
                                )
                            else:
                                flash(
                                    "Document stored and indexed successfully.",
                                    "success",
                                )
                        else:
                            flash(
                                "Document saved locally. Configure OPENAI_API_KEY to enable indexing.",
                                "error",
                            )
                        return redirect(url_for("docs"))

    base_query = Document.query.filter_by(user_id=user_id)
    doc_count = base_query.count()
    page = max(request.args.get("page", 1, type=int), 1)
    docs_query = base_query.order_by(Document.created_at.desc())
    docs = docs_query.limit(DOCS_PAGE_SIZE + 1).offset((page - 1) * DOCS_PAGE_SIZE).all()
    has_next = len(docs) > DOCS_PAGE_SIZE
    if has_next:
        docs = docs[:-1]
    has_prev = page > 1

    return render_template(
        "docs.html",
        docs=docs,
        page=page,
        has_next=has_next,
        has_prev=has_prev,
        doc_count=doc_count,
        doc_limit=DOCUMENT_LIMIT_PER_USER,
        username=session["username"],
        doc_error=doc_error,
    )


@app.route("/search", methods=["GET", "POST"])
def search():
    if "user_id" not in session:
        return redirect(url_for("login"))

    query_text = ""
    top_k = DEFAULT_RETRIEVAL_K
    search_error = None
    results: List[Dict[str, Any]] = []
    trigger = False

    if request.method == "POST":
        query_text = request.form.get("query", "").strip()
        top_k = _safe_k(request.form.get("k"))
        trigger = True
    else:
        query_text = request.args.get("q", "").strip()
        top_k = _safe_k(request.args.get("k"))
        trigger = bool(query_text)

    retrieval_available = _openai_available()

    if trigger and not query_text:
        search_error = "Enter a query to search your knowledge base."
    elif query_text:
        if not retrieval_available:
            search_error = "Retrieval is unavailable until OPENAI_API_KEY is configured."
        else:
            try:
                results = retrieve_documents_for_user(session["user_id"], query_text, top_k)
            except Exception:
                current_app.logger.exception("Vector search failed.")
                search_error = "Search failed. Please try again."

    return render_template(
        "search.html",
        username=session["username"],
        query=query_text,
        top_k=top_k,
        results=results,
        search_error=search_error,
        retrieval_available=retrieval_available,
    )


@app.route("/judge", methods=["POST"])
def judge():
    if "user_id" not in session:
        return redirect(url_for("login"))

    doc_id = request.form.get("doc_id", type=int)
    query_text = request.form.get("query", "").strip()
    if not doc_id or not query_text:
        return redirect(url_for("search"))

    doc = Document.query.get(doc_id)
    if not doc or doc.user_id != session["user_id"]:
        return redirect(url_for("search"))

    binary = _parse_binary(request.form.get("binary"))
    grade = _parse_int(request.form.get("grade"))
    rank = _parse_int(request.form.get("rank"))

    try:
        judgment = Judgment(
            user_id=session["user_id"],
            query_text=query_text,
            doc_id=doc_id,
            binary=binary,
            grade=grade,
            rank=rank,
        )
        db.session.add(judgment)
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Failed to save judgment.")

    return redirect(url_for("search", q=query_text, k=request.form.get("k", DEFAULT_RETRIEVAL_K)))


@app.route("/eval")
def eval_view():
    if "user_id" not in session:
        return redirect(url_for("login"))
    metrics = compute_metrics(session["user_id"], DEFAULT_RETRIEVAL_K)
    return render_template(
        "eval.html",
        metrics=metrics,
        username=session["username"],
    )


# ----------------------------
# Image Generation + Gallery
# ----------------------------
@app.route("/index", methods=["GET", "POST"])
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))

    context: Dict[str, Any] = {
        "username": session["username"],
        "retrieved_docs": [],
        "rag_enabled": False,
        "retrieval_error": None,
        "error": None,
        "share_prompt": False,
    }
    pending_result = session.pop(GENERATION_SESSION_KEY, None)
    if pending_result:
        image = Image.query.get(pending_result.get("image_id"))
        if image and image.user_id == session["user_id"]:
            context["image"] = image
            context["optimized_prompt"] = pending_result.get("optimized_prompt", image.prompt)
            serialized = pending_result.get("retrieved_docs", [])
            context["retrieved_docs"] = _hydrate_retrieval(serialized, session["user_id"])
            context["rag_enabled"] = pending_result.get("rag_enabled", False)
            context["share_prompt"] = pending_result.get("share_prompt", False)

    if request.method == "POST":
        user_prompt = request.form.get("prompt", "").strip()
        use_rag = bool(request.form.get("use_rag"))
        share_prompt = bool(request.form.get("share_prompt"))
        context["rag_enabled"] = use_rag
        context["share_prompt"] = share_prompt

        if not user_prompt:
            context["error"] = "Prompt cannot be empty."
            return render_template("index.html", **context)

        client: OpenAI | None = current_app.config.get("OPENAI_CLIENT")
        if client is None:
            context["error"] = "Image generation requires OPENAI_API_KEY to be configured."
            return render_template("index.html", **context)

        retrieved_docs: List[Dict[str, Any]] = []
        serialized_retrieval: List[Dict[str, Any]] = []
        if use_rag:
            try:
                retrieved_docs = retrieve_documents_for_user(
                    session["user_id"], user_prompt, DEFAULT_RETRIEVAL_K
                )
                serialized_retrieval = _serialize_retrieval(retrieved_docs)
            except RuntimeError as exc:
                current_app.logger.warning("Retrieval unavailable: %s", exc)
                context["retrieval_error"] = str(exc)
            except Exception:
                current_app.logger.exception("Failed to retrieve context snippets.")
                context["retrieval_error"] = "Retrieval failed; continuing without context."
            context["retrieved_docs"] = retrieved_docs

        optimized_prompt = (
            optimize_with_context(user_prompt, [doc["text"] for doc in retrieved_docs])
            if use_rag and retrieved_docs
            else optimize_prompt(user_prompt)
        )
        context["optimized_prompt"] = optimized_prompt

        filename = f"{uuid.uuid4().hex}.png"
        filepath: Path | None = None

        try:
            result = client.images.generate(
                model=IMAGE_MODEL,
                prompt=optimized_prompt,
                size="1024x1024",
            )
            image_data = result.data[0]
            filepath = Path(current_app.config["UPLOAD_FOLDER"]) / filename

            image_url = getattr(image_data, "url", None)
            if image_url:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                filepath.write_bytes(response.content)
            else:
                image_b64 = getattr(image_data, "b64_json", None)
                if not image_b64:
                    raise ValueError("Image generation response missing both URL and data.")
                filepath.write_bytes(base64.b64decode(image_b64))

            enhance_image(str(filepath))

            new_image = Image(
                filename=filename,
                prompt=optimized_prompt,
                timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                user_id=session["user_id"],
            )

            rag_doc = None
            should_store_prompt = (use_rag or share_prompt) and optimized_prompt
            if should_store_prompt:
                rag_hash = _hash_text(optimized_prompt)
                doc_total = Document.query.filter_by(user_id=session["user_id"]).count()
                if doc_total >= DOCUMENT_LIMIT_PER_USER:
                    flash(
                        "Document limit reached; new prompts were not saved to your knowledge base.",
                        "error",
                    )
                else:
                    existing_doc = Document.query.filter_by(
                        user_id=session["user_id"], text_hash=rag_hash
                    ).first()
                    if existing_doc:
                        if share_prompt and not existing_doc.is_public:
                            existing_doc.is_public = True
                    else:
                        rag_doc = Document(
                            text=optimized_prompt,
                            user_id=session["user_id"],
                            text_hash=rag_hash,
                            is_public=share_prompt,
                        )
                        db.session.add(rag_doc)

            db.session.add(new_image)
            db.session.flush()

            doc_id_for_index = rag_doc.id if rag_doc else None
            db.session.commit()

            if doc_id_for_index:
                try:
                    add_document_to_index(doc_id_for_index, optimized_prompt)
                except Exception:
                    current_app.logger.exception("Failed to index optimized prompt.")

            session[GENERATION_SESSION_KEY] = {
                "image_id": new_image.id,
                "optimized_prompt": optimized_prompt,
                "retrieved_docs": serialized_retrieval,
                "rag_enabled": use_rag,
                "share_prompt": share_prompt,
            }
            session.modified = True
            return redirect(url_for("index"))

        except BadRequestError as exc:
            if filepath and filepath.exists():
                filepath.unlink(missing_ok=True)
            db.session.rollback()
            current_app.logger.warning("OpenAI safety filter triggered: %s", exc)
            context["error"] = _extract_safety_message(exc)
            return render_template("index.html", **context)
        except requests.RequestException:
            if filepath and filepath.exists():
                filepath.unlink(missing_ok=True)
            db.session.rollback()
            current_app.logger.exception("Image download failed.")
            context["error"] = "Image download failed. Please try again."
            return render_template("index.html", **context)
        except Exception as exc:
            if filepath and filepath.exists():
                filepath.unlink(missing_ok=True)
            db.session.rollback()
            current_app.logger.exception("Image generation failed.")
            context["error"] = str(exc)
            return render_template("index.html", **context)

    return render_template("index.html", **context)


@app.route("/gallery")
def gallery():
    if "user_id" not in session:
        return redirect(url_for("login"))
    page = max(request.args.get("page", 1, type=int), 1)
    query = Image.query.filter_by(user_id=session["user_id"]).order_by(Image.id.desc())
    images = query.limit(GALLERY_PAGE_SIZE + 1).offset((page - 1) * GALLERY_PAGE_SIZE).all()
    has_next = len(images) > GALLERY_PAGE_SIZE
    if has_next:
        images = images[:-1]
    has_prev = page > 1
    return render_template(
        "gallery.html",
        images=images,
        username=session["username"],
        page=page,
        has_next=has_next,
        has_prev=has_prev,
    )


@app.route("/images/<int:image_id>")
def serve_image(image_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    image = Image.query.get_or_404(image_id)
    if image.user_id != session["user_id"]:
        abort(404)
    file_path = Path(current_app.config["UPLOAD_FOLDER"]) / image.filename
    if not file_path.exists():
        legacy_path = Path(app.root_path) / "static" / "images" / image.filename
        if legacy_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.replace(file_path)
        else:
            abort(404)
    download = request.args.get("download") == "1"
    return send_from_directory(
        current_app.config["UPLOAD_FOLDER"],
        image.filename,
        as_attachment=download,
    )


# ----------------------------
# Evaluation Helpers
# ----------------------------
def compute_metrics(user_id: int, k: int = DEFAULT_RETRIEVAL_K) -> Dict[str, Any]:
    """Compute Precision@k, MRR, and nDCG@k for a user's judgments."""
    judgments = (
        Judgment.query.filter_by(user_id=user_id)
        .order_by(Judgment.query_text.asc(), Judgment.rank.asc(), Judgment.created_at.asc())
        .all()
    )
    if not judgments:
        return {
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "query_count": 0,
            "judgment_count": 0,
        }

    grouped: Dict[str, List[Judgment]] = {}
    for item in judgments:
        grouped.setdefault(item.query_text, []).append(item)

    precisions: List[float] = []
    reciprocal_ranks: List[float] = []
    ndcgs: List[float] = []

    for items in grouped.values():
        items.sort(key=lambda j: ((j.rank if j.rank is not None else math.inf), j.created_at))
        top_items = items[:k]
        if not top_items:
            continue

        denom = min(k, len(top_items)) or 1
        relevant_count = sum(1 for j in top_items if j.binary is True)
        precisions.append(relevant_count / denom)

        rr = 0.0
        for idx, judgment in enumerate(top_items, start=1):
            if judgment.binary is True:
                rr = 1.0 / idx
                break
        reciprocal_ranks.append(rr)

        gains = [max(judgment.grade or 0, 0) for judgment in top_items]
        if any(gains):
            ideal_gains = sorted(gains, reverse=True)
            ndcg = _dcg(gains) / (_dcg(ideal_gains) or 1.0)
            ndcgs.append(ndcg)
        else:
            ndcgs.append(0.0)

    return {
        "precision_at_k": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "query_count": len(grouped),
        "judgment_count": len(judgments),
    }


def _dcg(gains: Sequence[float]) -> float:
    total = 0.0
    for idx, gain in enumerate(gains, start=1):
        total += (2**gain - 1) / math.log2(idx + 1)
    return total


def _safe_k(value: str | None) -> int:
    try:
        parsed = int(value) if value is not None else DEFAULT_RETRIEVAL_K
    except ValueError:
        parsed = DEFAULT_RETRIEVAL_K
    return max(1, min(parsed, 10))


def _parse_binary(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value == "1"


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_safety_message(exc: BadRequestError) -> str:
    message = str(exc)
    lowered = message.lower()
    if "safety" in lowered or "policy" in lowered:
        return "OpenAI safety filters blocked this prompt. Please adjust your wording."
    return "OpenAI rejected this prompt. Please try again with a different description."


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    debug_flag = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug_flag)


# ----------------------------
# Usage Instructions
# ----------------------------
"""
Quickstart:
1. python -m venv .venv && .venv\\Scripts\\activate
2. pip install -r requirements.txt
3. flask --app app init-db
4. set OPENAI_API_KEY=your_key_here (PowerShell: $Env:OPENAI_API_KEY="...")
5. flask --app app run
Workflow: Register -> Docs -> Search/Judge -> Generate (RAG) -> Evaluate.
"""
