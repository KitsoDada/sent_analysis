import csv
import hmac
import io
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

from src.review_service import analyze_review, create_reply_draft

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "app" / "reviews.sqlite3"


def get_db(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_db(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with get_db(path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT UNIQUE,
                product TEXT NOT NULL DEFAULT '',
                rating INTEGER,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                sentiment TEXT NOT NULL,
                themes TEXT NOT NULL,
                suggested_reply TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                approved_reply TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )"""
        )


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE=os.environ.get("REVIEWS_DATABASE", str(DEFAULT_DB)),
        REVIEW_API_KEY=os.environ.get("REVIEW_API_KEY", ""),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    )
    if test_config:
        app.config.update(test_config)
    initialize_db(app.config["DATABASE"])

    def authorized():
        expected = app.config["REVIEW_API_KEY"]
        supplied = request.headers.get("X-API-Key", "")
        return bool(expected) and hmac.compare_digest(expected, supplied)

    def require_auth():
        if not app.config["REVIEW_API_KEY"]:
            return jsonify(error="Configure REVIEW_API_KEY before using write endpoints."), 503
        if not authorized():
            return jsonify(error="Unauthorized"), 401
        return None

    def save_review(data, source="manual"):
        text = str(data.get("text", "")).strip()
        if not text:
            raise ValueError("Review text is required.")
        if len(text) > 10000:
            raise ValueError("Review text must be 10,000 characters or fewer.")
        rating = data.get("rating")
        if rating not in (None, ""):
            rating = int(rating)
            if not 1 <= rating <= 5:
                raise ValueError("Rating must be between 1 and 5.")
        analysis = analyze_review(text, rating=rating)
        review = {
            "external_id": str(data.get("external_id", "")).strip() or None,
            "product": str(data.get("product", ""))[:250],
            "rating": rating,
            "text": text,
            "source": str(data.get("source", source))[:80],
            "sentiment": analysis["sentiment"],
            "themes": ",".join(analysis["themes"]),
            "suggested_reply": create_reply_draft(analysis, text),
            "status": "pending",
            "approved_reply": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        fields = ", ".join(review)
        placeholders = ", ".join("?" for _ in review)
        updates = ", ".join(
            f"{key}=excluded.{key}" for key in review if key != "external_id"
        )
        with get_db(app.config["DATABASE"]) as connection:
            cursor = connection.execute(
                f"INSERT INTO reviews ({fields}) VALUES ({placeholders}) "
                f"ON CONFLICT(external_id) DO UPDATE SET {updates} "
                if review["external_id"]
                else f"INSERT INTO reviews ({fields}) VALUES ({placeholders})",
                tuple(review.values()),
            )
            review_id = cursor.lastrowid
            if review["external_id"]:
                row = connection.execute(
                    "SELECT id FROM reviews WHERE external_id=?", (review["external_id"],)
                ).fetchone()
                review_id = row["id"]
        return {"id": review_id, **review, "themes": analysis["themes"]}

    @app.get("/health")
    def health():
        return jsonify(status="ok", service="review-management-api")

    @app.post("/api/reviews/analyze")
    def analyze_endpoint():
        denied = require_auth()
        if denied:
            return denied
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify(error="Review text is required."), 400
        try:
            result = analyze_review(text, rating=data.get("rating"))
            return jsonify(
                **result,
                suggested_reply=create_reply_draft(result, text),
                requires_approval=True,
            )
        except (TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400

    @app.post("/api/reviews")
    def create_review():
        denied = require_auth()
        if denied:
            return denied
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(save_review(data)), 201
        except (TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400

    @app.post("/api/integrations/reviews")
    def ingest_reviews():
        """Provider adapter endpoint: submit normalized review objects in a batch."""
        denied = require_auth()
        if denied:
            return denied
        data = request.get_json(silent=True) or {}
        reviews = data.get("reviews")
        if not isinstance(reviews, list) or not reviews or len(reviews) > 100:
            return jsonify(error="Provide 1 to 100 reviews in a 'reviews' array."), 400
        results = []
        try:
            for review in reviews:
                if not isinstance(review, dict):
                    raise ValueError("Each review must be a JSON object.")
                results.append(save_review(review, source=data.get("source", "shopify_review_app")))
        except (TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400
        return jsonify(imported=len(results), reviews=results), 201

    @app.get("/api/reviews")
    def list_reviews():
        denied = require_auth()
        if denied:
            return denied
        status = request.args.get("status")
        sentiment = request.args.get("sentiment")
        query = "SELECT * FROM reviews WHERE 1=1"
        params = []
        if status:
            query += " AND status=?"
            params.append(status)
        if sentiment:
            query += " AND sentiment=?"
            params.append(sentiment)
        query += " ORDER BY created_at DESC LIMIT 500"
        with get_db(app.config["DATABASE"]) as connection:
            rows = connection.execute(query, params).fetchall()
        return jsonify(reviews=[dict(row, themes=row["themes"].split(",") if row["themes"] else []) for row in rows])

    @app.patch("/api/reviews/<int:review_id>")
    def update_review(review_id):
        denied = require_auth()
        if denied:
            return denied
        data = request.get_json(silent=True) or {}
        status = data.get("status")
        reply = data.get("approved_reply")
        if status not in ("pending", "approved", "dismissed"):
            return jsonify(error="Status must be pending, approved, or dismissed."), 400
        if status == "approved" and not str(reply or "").strip():
            return jsonify(error="An edited or suggested reply is required for approval."), 400
        with get_db(app.config["DATABASE"]) as connection:
            cursor = connection.execute(
                "UPDATE reviews SET status=?, approved_reply=? WHERE id=?",
                (status, str(reply or "").strip(), review_id),
            )
        if not cursor.rowcount:
            return jsonify(error="Review not found."), 404
        return jsonify(id=review_id, status=status, approved_reply=str(reply or "").strip())

    @app.post("/api/reviews/import-csv")
    def import_csv():
        denied = require_auth()
        if denied:
            return denied
        upload = request.files.get("file")
        if not upload or not upload.filename.lower().endswith(".csv"):
            return jsonify(error="Upload a .csv file with a text column."), 400
        try:
            rows = csv.DictReader(io.StringIO(upload.stream.read().decode("utf-8-sig")))
            if not rows.fieldnames or "text" not in rows.fieldnames:
                return jsonify(error="CSV must contain a 'text' column."), 400
            imported = [save_review(row, source="csv_import") for row in rows]
            return jsonify(imported=len(imported), reviews=imported), 201
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            return jsonify(error=f"CSV import failed: {error}"), 400

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
