"""
Database connection setup.

Uses SQLAlchemy. DATABASE_URL comes from settings (app/core/config.py) —
SQLite for local dev, swap to Postgres/MySQL in production by just
changing that one env var.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session, closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
