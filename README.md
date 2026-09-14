# Styla

**AI Personal Stylist — Wardrobe-Aware Outfit Recommendation with Reference-Look Matching**

Styla is an AI-powered personal stylist app that analyzes a user's own clothes and suggests outfit combinations. Users upload photos of their garments, and the system extracts their attributes and generates outfit combinations. Users can also upload a photo of any outfit they like to see how closely it can be recreated with their own wardrobe, and receive product suggestions for whatever pieces are missing.

This is a 5-person ML final project built for [course name].

---

## Contents

- [Problem](#problem)
- [How it works](#how-it-works)
- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Datasets](#datasets)
- [Evaluation metrics](#evaluation-metrics)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Team](#team)
- [Timeline](#timeline)
- [Future work](#future-work)

---

## Problem

People often don't know what outfits they can put together from clothes they already own, and struggle to recreate outfits they like using their own wardrobe. Deciding what to wear takes time, and when people can't combine what they have, they tend to default to the same few outfits, leaving the rest of their wardrobe unused — or they buy new clothes they don't actually need.

## How it works

1. The user uploads a photo of a garment → the system removes the background, detects the item, and extracts attributes such as category, color, and pattern
2. Each garment gets a vector representation (embedding) and is stored in the wardrobe database
3. A compatibility model generates outfit combinations from the items in the user's wardrobe
4. If the user uploads a photo of an outfit they like, the system parses it into individual items, matches each one against their own wardrobe, and identifies what's missing
5. For each missing item, the system suggests alternatives from a static product catalog

See [Architecture](#architecture) for the full pipeline diagram.

## Features

**MVP (in progress this sprint cycle):**

- Adding garments via photo and building a wardrobe
- Automatic attribute recognition (category, color, pattern)
- Outfit generation from the user's own wardrobe
- Reference outfit analysis and matching against the wardrobe
- Detection of missing items with alternatives suggested from a static catalog
- Simple personalization (rerank) based on a style quiz

**Future work (post-MVP):**

- Learning recommendations over time from user feedback (deferred due to cold-start — no user data yet)
- Live store integration (replacing the static catalog)
- B2B model — stores and brands integrating their own catalogs with the system

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: User's garment photo   |   Reference outfit photo    │
└───────────────┬──────────────────────┬──────────────────────┘
                │                      │
                ▼                      ▼
      ┌─────────────────┐    ┌──────────────────────┐
      │ Background       │    │ Outfit parsing       │
      │ removal (rembg)  │    │ (split into items)   │
      └────────┬────────┘    └──────────┬───────────┘
               │                        │
               ▼                        │
      ┌─────────────────┐               │
      │ Attribute model │               │
      │ (category,      │               │
      │ color, pattern) │               │
      └────────┬────────┘               │
               │                        │
               ▼                        ▼
      ┌──────────────────────────────────────────┐
      │  EMBEDDING (FashionCLIP)                 │
      │  image → vector                          │
      └────────┬─────────────────────────────────┘
               │
               ▼
      ┌──────────────────────────────────────────┐
      │  PostgreSQL + pgvector                   │
      │  wardrobe · catalog · user profile       │
      └────────┬─────────────────────────────────┘
               │
     ┌─────────┼──────────────┬───────────────────┐
     ▼         ▼              ▼                   ▼
┌─────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐
│Compat.  │ │ Outfit   │ │  Reference   │ │  Catalog     │
│ Scorer  │→│Generator │ │  Matcher     │ │  search      │
└─────────┘ └────┬─────┘ └──────┬───────┘ └──────┬───────┘
                 │              │                │
                 └──────────────┼────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │  Style quiz rerank     │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │  FastAPI → React UI    │
                    └────────────────────────┘
```

## Tech stack

| Category | Tools |
|---|---|
| ML / CV | Python, PyTorch, FashionCLIP, rembg |
| Vector search | PostgreSQL + pgvector |
| Backend | FastAPI |
| Frontend | React |
| Experiment tracking | Weights & Biases (or MLflow) |
| Containerization | Docker, Docker Compose |
| Version control | Git / GitHub |

## Datasets

| Dataset | Purpose |
|---|---|
| [Polyvore Outfits](https://github.com/xthan/polyvore-dataset) | Outfit compatibility (primary dataset) |
| [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) | Detection, segmentation |
| [Fashionpedia](https://fashionpedia.github.io/home/Fashionpedia_download.html) | Detailed garment attributes |
| Our own real-world test set | ~100 phone photos from team members' own wardrobes — used to measure the domain gap between clean dataset images and real user photos |

> Note: Polyvore's official site shut down in 2018; the dataset is distributed via GitHub mirrors. If it's unavailable, fallback plan: Fashionpedia + a small hand-curated outfit set.

## Evaluation metrics

| Component | Metric | Target |
|---|---|---|
| Attribute recognition | macro-F1 (val / real test) | ≥ 0.80 / ≥ 0.65 |
| Compatibility | AUC, FITB accuracy | ≥ 0.80 |
| Reference matching | Recall@5 | ≥ 0.7 |
| Outfit quality | Human eval (1–5) | ≥ 3.5 |
| Latency | top-5 outfit generation | < 2 seconds |

Baseline chain: **random selection → CLIP cosine similarity → trained model**. Every experiment is logged in W&B.

## Project structure

```
styla/
├── backend/            # FastAPI app, API endpoints
├── frontend/           # React app
├── ml/
│   ├── data/           # dataset download/cleaning scripts
│   ├── vision/         # segmentation, attribute model
│   ├── compatibility/  # compatibility model, scorer
│   ├── style/          # ensemble-level style classifier
│   ├── retrieval/       # embedding, pgvector integration
│   └── evaluate.py     # metrics: accuracy, F1, AUC, FITB, Recall@k
├── data/
│   ├── raw/            # downloaded datasets (git-ignored)
│   └── test-set/       # the team's own labeled real-world photos
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

### 1. Database

One PostgreSQL database with the [pgvector](https://github.com/pgvector/pgvector) extension holds everything: accounts, wardrobe embeddings, personal-style references and saved looks. Point `DATABASE_URL` at it in `.env`:

- **Local** — `docker compose up -d` starts a pgvector container on port 5440 (`init.sql` creates the schema).
- **Shared** — the team's Neon Postgres (pooled connection string with `sslmode=require`).

The backend applies the schema on every start (`backend/db.py`), so switching databases needs no manual migration. `python -m backend.db` runs the migration standalone and prints a status.

### 2. Backend

```bash
cp .env.example .env            # then set DATABASE_URL (and optionally STYLA_JWT_SECRET)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --port 8000
```

On first start a demo account is created (`demo@styla.app` / `demo1234`, configurable via `STYLA_DEMO_*`). Any wardrobe rows that pre-date accounts are assigned to it.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173 (VITE_API_URL defaults to http://localhost:8000)
```

`run.bat` does all three steps on Windows.

### 4. Seed a wardrobe (optional)

```bash
python seed_hf.py --count 60                      # HuggingFace fashion product photos -> demo account
python seed_db.py --email you@example.com --password ...   # Platzi store photos -> your account
```

## Accounts & API

Sign up / sign in from the app (`/signup`, `/login`). Sessions are JWTs (7 days) sent as `Authorization: Bearer <token>`; passwords are bcrypt-hashed. Every endpoint below is scoped to the signed-in user.

| Endpoint | Purpose |
|---|---|
| `POST /api/auth/signup`, `POST /api/auth/login`, `GET /api/auth/me` | accounts |
| `GET/POST /api/wardrobe`, `DELETE /api/wardrobe/{id}` | garments (upload = background removal + FashionCLIP attributes + embedding) |
| `POST /api/generate` | outfits for a style (`backend/outfits.py`): candidate top+bottom / dress (+ outerwear) combos scored on compatibility, colour harmony, zero-shot style similarity, personal-style similarity and style rules, then a diverse top-N. Options: `outerwear` (`auto`/`always`/`never`), `must_include` (build every look around one garment), `exclude_ids`, and `offset` (skip looks already shown, so asking again returns new combinations) |
| `POST /api/reference/match` | reference look (`backend/reference.py`): clothes segmentation splits the photo into top / bottom / dress crops, each crop is embedded and searched against the user's items in the same slot; missing pieces get shop-search links |
| `GET/POST /api/looks`, `DELETE /api/looks/{id}` | saved looks |
| `GET/POST/DELETE /api/style/personal` | personal-style reference photos |
| `POST /api/tryon` | virtual try-on: base layer, then bottom, then jacket, each fed into the next step. Each garment is tried against several Hugging Face Spaces in turn (IDM-VTON, OOTDiffusion, CatVTON); if all are out of GPU quota or asleep it falls back to an offline preview (`ml/vton/local_preview.py`) that draws the garments onto the photo locally, so try-on always returns something. The response's `mode` is `ai` or `preview`, and `applied`/`skipped` report per-garment outcomes. Set `HF_TOKEN` for a larger GPU quota |

## Team

| | Person | Technical area |
|---|---|---|
| **A** |  | Compatibility & Ranking — compatibility model, scorer, baseline comparisons |
| **B** |  | Computer Vision — segmentation, detection, attribute classification |
| **C** |  | Representation & Retrieval — embeddings, vector search, reference matching |
| **D** |  | Data & Evaluation — datasets, evaluation harness, metrics, reporting |
| **E** |  | Application — FastAPI/React, outfit generator, catalog |

Roles are a technical split, not a hierarchy — there's no formal lead, decisions are made by consensus.

## Timeline

| Phase | Dates |
|---|---|
| Research and dataset | Aug 17–22 |
| End-to-end skeleton | Aug 17–24 |
| Wardrobe representation | Aug 22 – Sep 2 |
| Compatibility model | Aug 26 – Sep 4 |
| Recommendation | Sep 2–8 |
| Reference matching | Sep 2–8 |
| Shopping & personalization | Sep 3–9 |
| Final (freeze, report, demo) | Sep 9–15 |


## Future work

- Personalization that improves over time from user feedback (once real user data exists)
- Live store scraping / retailer API integrations
- Mobile app version
- B2B: stores and brands connecting their own catalogs to the system

---

*This README will be updated as the project evolves.*