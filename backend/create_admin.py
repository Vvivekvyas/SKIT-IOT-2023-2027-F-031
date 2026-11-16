"""
One-time script to create the first admin user.
Run from inside the backend/ folder:  python create_admin.py
"""
import getpass

from app.core.security import hash_password
from app.db.database import get_db
from app.models.user import User

username = input("Admin username: ").strip()
password = getpass.getpass("Admin password (min 8 chars): ")

if len(username) < 3 or len(password) < 8:
    raise SystemExit("Username min 3 chars, password min 8 chars.")

db = next(get_db())
try:
    if db.query(User).filter(User.username == username).first():
        raise SystemExit("That username already exists.")

    db.add(User(username=username, hashed_password=hash_password(password), role="admin"))
    db.commit()
    print(f"Admin '{username}' created.")
finally:
    db.close()