import os

import streamlit as st

from src.review_service import analyze_review, create_reply_draft
from app.review_api import get_db, initialize_db

DB_PATH = os.environ.get("REVIEWS_DATABASE", str(__import__("pathlib").Path(__file__).resolve().parent / "reviews.sqlite3"))
initialize_db(DB_PATH)
st.set_page_config(page_title="Review Desk", page_icon="🛍️", layout="wide")
st.title("🛍️ Ecommerce Review Desk")
st.caption("Analyze, organize, and draft replies. Replies stay pending until a staff member approves them.")

with st.sidebar:
    st.header("Import reviews")
    uploaded = st.file_uploader("Upload review CSV", type=["csv"])
    if uploaded and st.button("Import CSV"):
        try:
            import pandas as pd
            rows = pd.read_csv(uploaded).fillna("").to_dict(orient="records")
            count = 0
            for row in rows:
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                rating = row.get("rating") or None
                analysis = analyze_review(text, rating=rating)
                reply = create_reply_draft(analysis, text)
                with get_db(DB_PATH) as connection:
                    connection.execute(
                        "INSERT INTO reviews (external_id, product, rating, text, source, sentiment, themes, suggested_reply, status, approved_reply, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', '', datetime('now'))",
                        (str(row.get("external_id") or "") or None, str(row.get("product", "")), int(rating) if rating else None, text, "csv_import", analysis["sentiment"], ",".join(analysis["themes"]), reply),
                    )
                count += 1
            st.success(f"Imported {count} reviews.")
        except Exception as error:
            st.error(f"Import failed: {error}")
    st.markdown("**Shopify connection**")
    st.caption("Connect a Shopify review app/provider through the normalized API endpoint. Configure its approved API/webhook credentials on the server; do not paste access tokens here.")
    st.code("POST /api/integrations/reviews", language="text")

filters = st.columns(3)
with filters[0]:
    sentiment_filter = st.selectbox("Sentiment", ["all", "positive", "neutral", "negative"])
with filters[1]:
    status_filter = st.selectbox("Status", ["all", "pending", "approved", "dismissed"])
with filters[2]:
    search = st.text_input("Search reviews or products")

query = "SELECT * FROM reviews WHERE 1=1"
params = []
if sentiment_filter != "all":
    query += " AND sentiment=?"
    params.append(sentiment_filter)
if status_filter != "all":
    query += " AND status=?"
    params.append(status_filter)
query += " ORDER BY created_at DESC LIMIT 500"
with get_db(DB_PATH) as connection:
    records = [dict(row) for row in connection.execute(query, params).fetchall()]

if search:
    needle = search.casefold()
    records = [r for r in records if needle in r["text"].casefold() or needle in r["product"].casefold()]

summary = st.columns(4)
summary[0].metric("Reviews", len(records))
summary[1].metric("Negative", sum(r["sentiment"] == "negative" for r in records))
summary[2].metric("Pending approval", sum(r["status"] == "pending" for r in records))
summary[3].metric("Themes detected", len({theme for r in records for theme in r["themes"].split(",") if theme}))

if not records:
    st.info("No reviews yet. Import a CSV with a `text` column, or submit reviews to the API.")

for review in records:
    title = f"#{review['id']} · {review['sentiment'].title()} · {review['product'] or 'Review'} · {review['status']}"
    with st.expander(title):
        st.write(review["text"])
        st.caption(f"Source: {review['source']} · Rating: {review['rating'] or 'not supplied'} · Themes: {review['themes'] or 'general'}")
        st.markdown("**Suggested reply (draft)**")
        draft = st.text_area("Edit before approval", value=review["suggested_reply"], key=f"reply-{review['id']}")
        left, right = st.columns(2)
        if left.button("Approve reply", key=f"approve-{review['id']}", disabled=review["status"] == "approved"):
            if not draft.strip():
                st.error("Reply cannot be empty.")
            else:
                with get_db(DB_PATH) as connection:
                    connection.execute("UPDATE reviews SET status='approved', approved_reply=? WHERE id=?", (draft.strip(), review["id"]))
                st.success("Approved and saved. This does not publish to Shopify.")
                st.rerun()
        if right.button("Dismiss", key=f"dismiss-{review['id']}", disabled=review["status"] == "dismissed"):
            with get_db(DB_PATH) as connection:
                connection.execute("UPDATE reviews SET status='dismissed' WHERE id=?", (review["id"],))
            st.rerun()
