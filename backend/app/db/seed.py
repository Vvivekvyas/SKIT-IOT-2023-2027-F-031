"""
Run once to set up your local database:
    python -m app.db.seed

Creates all tables and inserts one test user, if it doesn't already exist.
"""
from app.core.security import hash_password
from app.db.database import Base, SessionLocal, engine
from app.models.dataset import Dataset  # noqa: F401 — imported so create_all sees it
from app.models.prediction import Prediction  # noqa: F401
from app.models.user import User


def seed():
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == "analyst@example.com").first()
        if not existing:
            test_user = User(
                username="analyst@example.com",
                hashed_password=hash_password("changeme123"),
                role="analyst",
            )
            db.add(test_user)
            db.commit()
            print("Created test user: analyst@example.com / changeme123")
        else:
            print("Test user already exists — nothing to do.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
