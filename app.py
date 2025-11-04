import base64
import datetime
import logging
import os
import secrets
import shutil
from pathlib import Path

import click
import requests
from flask import Flask, current_app, redirect, render_template, request, session, url_for
from flask.cli import with_appcontext
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI
from sqlalchemy import String, inspect
from werkzeug.security import check_password_hash, generate_password_hash

from utils.image_enhancer import enhance_image
from utils.prompt_optimizer import optimize_prompt

# ----------------------------
# App Configuration
# ----------------------------
db = SQLAlchemy()
_schema_bootstrapped = False

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

upload_folder = Path(app.root_path) / "static" / "images"
upload_folder.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(upload_folder)

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is required to run the application."
    )
app.config["OPENAI_CLIENT"] = OpenAI(api_key=openai_api_key)

db.init_app(app)

# ----------------------------
# Database Models
# ----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    images = db.relationship("Image", backref="user", lazy=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


def ensure_database_schema(force: bool = False) -> None:
    """Create or upgrade database tables as needed."""
    engine = db.engine
    inspector = inspect(engine)

    existing_tables = set(inspector.get_table_names())
    if force or {"user", "image"} - existing_tables:
        db.create_all()
        inspector = inspect(engine)

    if "image" not in inspector.get_table_names():
        return

    if _image_prompt_column_needs_upgrade(inspector):
        _upgrade_image_prompt_column(engine)



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


@click.command("init-db")
@with_appcontext
def init_db_command() -> None:
    """Initialize or upgrade the application database."""
    ensure_database_schema(force=True)
    click.echo("Database initialized.")


def bootstrap_schema_if_needed() -> None:
    global _schema_bootstrapped
    if _schema_bootstrapped:
        return
    ensure_database_schema()
    _schema_bootstrapped = True


app.cli.add_command(init_db_command)


@app.before_request
def _ensure_schema() -> None:
    bootstrap_schema_if_needed()

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("index"))
    return redirect(url_for("login"))


# --- Register ---
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


# --- Login ---
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


# --- Logout ---
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --- Main Image Generation Page ---
@app.route("/index", methods=["GET", "POST"])
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        user_prompt = request.form["prompt"].strip()
        optimized = optimize_prompt(user_prompt)
        client = current_app.config["OPENAI_CLIENT"]

        filename = None
        filepath: Path | None = None
        try:
            # Generate image using Images API
            result = client.images.generate(
                model="gpt-image-1",
                prompt=optimized,
                size="1024x1024",  # valid sizes: 1024x1024, 1024x1536, 1536x1024, auto
            )

            image_data = result.data[0]
            image_url = getattr(image_data, "url", None)
            filename = (
                f"user{session['user_id']}"
                f"_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            )
            filepath = Path(current_app.config["UPLOAD_FOLDER"]) / filename

            if image_url:
                # Download the image
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                filepath.write_bytes(response.content)
            else:
                image_b64 = getattr(image_data, "b64_json", None)
                if not image_b64:
                    raise ValueError("Image generation response missing both URL and data.")
                image_bytes = base64.b64decode(image_b64)
                filepath.write_bytes(image_bytes)

            # Enhance the image
            enhance_image(str(filepath))

            # Save record in database
            new_image = Image(
                filename=filename,
                prompt=optimized,
                timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                user_id=session["user_id"],
            )
            db.session.add(new_image)
            db.session.commit()

            return render_template(
                "index.html",
                image=filename,
                optimized_prompt=optimized,
                username=session["username"]
            )

        except requests.RequestException:
            if filepath and filepath.exists():
                filepath.unlink(missing_ok=True)
            current_app.logger.exception("Image download failed")
            return render_template(
                "index.html",
                error="Image download failed. Please try again.",
                username=session["username"],
            )
        except Exception as e:
            if filepath and filepath.exists():
                filepath.unlink(missing_ok=True)
            db.session.rollback()
            current_app.logger.exception("Image generation failed")
            return render_template(
                "index.html",
                error=str(e),
                username=session["username"],
            )

    return render_template("index.html", username=session["username"])


# --- Gallery Page ---
@app.route("/gallery")
def gallery():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user_images = (
        Image.query.filter_by(user_id=session["user_id"])
        .order_by(Image.id.desc())
        .all()
    )
    return render_template("gallery.html", images=user_images, username=session["username"])


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
