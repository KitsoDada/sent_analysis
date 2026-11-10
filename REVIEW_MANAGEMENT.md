# Review management API and dashboard

This adds an initial, provider-neutral review management layer to the sentiment analysis project. It includes a Flask API, SQLite review queue, Streamlit dashboard, theme tagging, response drafts, and an explicit human approval step. **Approved replies are stored only; the app does not publish replies to Shopify.**

## Install and run

Install the existing dependencies plus the API dependency:

```bash
pip install -r requirements.txt
pip install -r requirements-review-management.txt
python -m src.train
```

Configure a secret before starting the API. All API review/data endpoints require `X-API-Key`; write endpoints return 503 when no key is configured.

```bash
export REVIEW_API_KEY='replace-with-a-long-random-secret'
python -m app.review_api
```

In another terminal, with the same database setting if customized:

```bash
streamlit run app/dashboard.py
```

Default database is `app/reviews.sqlite3`; set `REVIEWS_DATABASE` to change it. Do not commit credentials or production customer data.

## API

All endpoints except `GET /health` require header `X-API-Key: <REVIEW_API_KEY>`.

- `POST /api/reviews/analyze` JSON `{"text":"Delivery was late", "rating":2}` returns sentiment, confidence, detected themes, and a suggested reply draft.
- `POST /api/reviews` saves one review in `pending` state.
- `POST /api/integrations/reviews` accepts up to 100 normalized provider reviews: `{"source":"provider_name", "reviews":[{"external_id":"...", "product":"...", "rating":2, "text":"..."}]}`. Repeated external IDs are upserted.
- `GET /api/reviews?status=pending&sentiment=negative` lists reviews.
- `PATCH /api/reviews/<id>` accepts `{"status":"approved", "approved_reply":"edited reply"}` or `pending`/`dismissed`. Approval requires nonempty reply text and never posts it externally.
- `POST /api/reviews/import-csv` accepts multipart field `file`; CSV requires `text` and may include `product`, `rating`, `external_id`.

## Shopify connection boundary

Shopify product/review data often comes from a review app rather than a universal Shopify product-review endpoint. This project provides a normalized ingestion endpoint for an authorized provider adapter. A production connector still needs a specific review provider, its approved API/webhook, Shopify app OAuth/scopes where applicable, webhook verification, per-store data isolation, and deployment secrets. Do not send customer access tokens through the dashboard or expose this development API publicly without HTTPS, store-level authorization, rate limits, and production security review.

The included sample model/dataset is only a prototype. Validate sentiment and themes on a representative, consented dataset before business use. Theme detection currently uses transparent keyword rules and may miss context or sarcasm. Suggested responses are generic drafts and require staff review.
