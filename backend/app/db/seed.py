"""
Run once to set up your local database:
    python -m app.db.seed

Creates all tables and inserts two test users (analyst + admin),
skipping any that already exist. Safe to re-run.
"""
from app.core.security import hash_password
from app.db.database import Base, SessionLocal, engine
from app.models.dataset import Dataset  # noqa: F401 — imported so create_all sees it
from app.models.prediction import Prediction  # noqa: F401
from app.models.user import User

TEST_USERS = [
    ("analyst@example.com", "changeme123", "analyst"),
    ("admin@example.com", "changeme123", "admin"),
]


def seed():
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        for username, password, role in TEST_USERS:
            existing = db.query(User).filter(User.username == username).first()
            if existing:
                print(f"{username} already exists — skipping.")
                continue
            db.add(User(username=username, hashed_password=hash_password(password), role=role))
            db.commit()
            print(f"Created {role}: {username} / {password}")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
