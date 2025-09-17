from waitress import serve
from app import app  # import your Flask instance from app.py

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
