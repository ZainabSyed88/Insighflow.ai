"""Database initialization and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings
from src.database.models import Base
from typing import Generator
from pathlib import Path
import os


def _ensure_sqlite_dir():
    """Ensure the parent directory for the SQLite file exists.

    This avoids sqlite3.OperationalError: unable to open database file
    when the directory has not yet been created.
    """
    url = settings.database_url
    # Only handle file-based sqlite URLs
    if url.startswith("sqlite:"):
        # Support formats like sqlite:///./data/db.db or sqlite:///data/db.db
        # Extract the file path after the scheme
        parts = url.split("///", 1)
        if len(parts) == 2:
            db_path = parts[1]
        else:
            # fallback: remove 'sqlite://' prefix
            db_path = url.replace("sqlite://", "")

        # Normalize and create parent dir
        try:
            db_file = Path(db_path)
            parent = db_file.parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort: ignore failures here and let SQLAlchemy raise if unreachable
            pass


def init_db() -> None:
    """Initialize database tables."""
    _ensure_sqlite_dir()
    engine = create_engine(settings.database_url, echo=False)
    Base.metadata.create_all(bind=engine)


def get_engine():
    """Get SQLAlchemy engine."""
    _ensure_sqlite_dir()
    return create_engine(settings.database_url, echo=False)


def get_session_factory():
    """Get session factory."""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Generator[Session, None, None]:
    """Get database session (for dependency injection)."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
