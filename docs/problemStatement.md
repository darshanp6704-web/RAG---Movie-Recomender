# Movie Buddy — Problem Statement & Project Context

> **Subtitle:** A vibe-based movie and series finder for India

---

## 1. Overview

Build a Retrieval-Augmented Generation (RAG) chatbot (**Movie Buddy**) that recommends movies and series from a fixed catalog by **mood**, **plot**, and **structured filters**, and tells users **where they can legally watch each pick in India**.

---

## 2. Target Users

- College students in India
- Anyone who spends more time choosing what to watch than actually watching

---

## 3. Data Source & Constraints

The **only** data source for this project is the Excel workbook:
- **Problem statement reference path:** `data/movie_buddy_starter_3000.xlsx`
- **Current workspace file:** `movie_buddy_starter_3000 (1).xlsx` (3,000 catalog rows in `Titles`, 8,886 availability rows in `Where to watch`)

### Workbook Sheets
1. **`Titles`** — 3,000 movie and series records with metadata, vibe tags, ratings, streaming summary, URLs, and pre-built `search_text` for embeddings.
2. **`Where to watch`** — 8,886 India (`IN`) streaming/rental/purchase availability rows joined via `title_id`.
3. **`Data dictionary`** — Definitions, sources, and intended uses for all 46 columns across `Titles` (39) and `Where to watch` (7).
4. **`Sources and credits`** — Attribution, licensing, and freshness rules.

### Data Rules & Attribution
- **Catalog Source:** TMDB (`https://www.themoviedb.org`)
- **Streaming Availability Source:** JustWatch via TMDB (`https://www.justwatch.com`)
- **Embedding Model License:** `BAAI/bge-small-en-v1.5` (MIT license)
- **Strict Single-Source Rule:** No web scraping. No other external APIs for movie/series data.
- **Usage & Freshness:** Educational, non-commercial use only; do not sell this data. Legal platforms only (no piracy links). Streaming availability changes monthly — always check and display the `as_of` date.

---

## 4. Functional Requirements (The Assistant Must)

1. **Natural Language Recommendations:** Accept a natural language request and reply with **up to 5 picks**.
2. **Filter Understanding:** Extract and apply structured filters from the user's request:
   - Language (and industry/origin where relevant)
   - Type: Movie or Series
   - Era or release year
   - Length (runtime / length bucket / seasons / episodes / binge hours)
   - Platform (e.g., Netflix, Amazon Prime Video, JioHotstar, ZEE5, SonyLIV, aha, Sun NXT, YouTube, etc.)
   - Free to watch (`free_to_watch` / `Free` / `Free with ads`)
   - Certificate (`certification_in` — India ratings: `U`, `UA`, `A`, and TV equivalents)
   - Intensity (`Light`, `Medium`, `Heavy`)
   - Content to avoid (`content_notes`: Violence, Gore, Sexual content, Drugs, Suicide or self harm, Strong language, Adults only (A))
   - Who they are watching with (`watch_with`: Alone, Friends, Family, Date)
3. **Vibe & Semantic Understanding:** Understand mood, themes, plot descriptions, and similarity requests such as `"more like <title>"`.
4. **Required Pick Output Fields:** For every recommended pick, show:
   - Title
   - Year (`release_year`)
   - Language
   - Type (`Movie` or `Series`)
   - Runtime (for movies) or seasons (for series)
   - Rating
   - One-line hook (`hook`)
   - One sentence explaining why it matches the user's request
   - Where to watch in India with **access type** (`Subscription`, `Free`, `Free with ads`, `Rent`, `Buy`) and the **`as_of` date**
   - TMDB link (`tmdb_url`)
5. **Catalog Grounding & Empty-Result Handling:** Only recommend titles that exist in the dataset. If nothing matches, explicitly say so and suggest which filter(s) to relax.
6. **No Spoilers:** Describe titles using only the `hook` and `overview` (never reveal plot twists or endings).

---

## 5. Guardrails

- **Anti-Piracy:** Refuse piracy requests (downloads, torrents, free links to paid films) and offer legal Indian streaming/rental/purchase platforms instead.
- **Prompt Injection Defense:** Ignore any instructions inside user messages that attempt to override or modify system rules.
- **Topic Scope:** Stay strictly on topic — movies and series only.
- **No Hallucinations:** Never invent a platform, rating, year, or title not present in the dataset.
- **Missing Certificate Caution:** `certification_in` is often empty; treat empty values as **Unknown**, never as safe or unsafe.
- **Freshness Transparency:** Always surface the `as_of` date for streaming availability since India catalog availability changes monthly.

---

## 6. UI Requirements

- **Welcome message** introducing Movie Buddy
- **4 example prompts** to help users get started quickly
- **Chat box** for natural language queries
- **Results rendered as cards** displaying the required pick details (poster, metadata, hook, match reason, Indian watch options + access types + `as_of` date, and TMDB link)
- **Required Footer (verbatim):**
  > `This product uses the TMDB API but is not endorsed or certified by TMDB. Streaming availability by JustWatch.`

---

## 7. Technical Constraints

- **Runtime:** Python 3.11. Free tools only; no paid APIs.
- **Embeddings:** `FastEmbed` with `BAAI/bge-small-en-v1.5` (ONNX). **No PyTorch** and **no `sentence-transformers`** (too large for free-tier hosting RAM limits).
- **Vector Database:** `ChromaDB`, local and persistent.
- **LLM:** Groq free tier. API keys/secrets stored in `.env` and never committed to version control.
- **Hosting:** Render free web service (**512 MB RAM** constraint).

---

## 8. Success Criteria

- **Evaluation:** All **20 test questions** pass (Phase 8).
- **Deployment:** App is live on a public Render URL.
- **Latency:** Answers arrive in **under 5 seconds** once warm.

---

## 9. Data Dictionary — Columns We Will Use

Every column documented in the `Data dictionary` sheet of `movie_buddy_starter_3000 (1).xlsx` is listed below across both data sheets (**39 columns** in `Titles` and **7 columns** in `Where to watch`, **46 columns total**).

### 9.1 `Titles` Sheet (39 Columns)

| # | Column | Meaning | Source | Used For |
|---|---|---|---|---|
| 1 | `title_id` | Unique ID. `m_` for movies, `tv_` for series | Built from TMDB ID | Join key |
| 2 | `type` | `Movie` or `Series` | TMDB | Filter |
| 3 | `title` | English title | TMDB | Display |
| 4 | `original_title` | Title in its original language | TMDB | Display |
| 5 | `language` | Original language | TMDB | Filter |
| 6 | `industry` | Sandalwood, Bollywood, Kollywood, Tollywood, Mollywood, Hollywood and English, Korean, Japanese, Spanish, French | Derived from language | Filter |
| 7 | `origin_country` | Country codes | TMDB | Filter |
| 8 | `release_year` | Year of release or first episode | TMDB | Filter |
| 9 | `era` | Decade, e.g. `1990s`, `2010s`. The user chooses it, nothing is forced | Derived | Filter |
| 10 | `runtime_min` | Movie length, or average episode length for a series | TMDB | Filter |
| 11 | `length_bucket` | `Under 1.5 hrs`, `1.5 to 2 hrs`, `2 to 2.5 hrs`, `Over 2.5 hrs`, or episode length band | Derived | Filter |
| 12 | `seasons` | Number of seasons (series only) | TMDB | Filter |
| 13 | `episodes` | Number of episodes (series only) | TMDB | Filter |
| 14 | `binge_hours` | Episodes x episode length, in hours (series only) | Derived | Filter |
| 15 | `status` | `Ended`, `Returning Series`, `Canceled` (series only) | TMDB | Filter |
| 16 | `certification_in` | India certificate: `U`, `UA`, `A` and TV equivalents. Often empty: treat empty as Unknown, never as safe or unsafe. | TMDB | Filter and guardrail |
| 17 | `genres` | Genres | TMDB | Filter and vibe |
| 18 | `keywords` | Community tags like `friendship`, `coming of age`, `time loop` | TMDB | Vibe |
| 19 | `mood` | Up to 3 moods (AI hint for chips and filters; vibe search itself uses embeddings). Checked against genres. | Generated: embedding similarity + genre hints | Filter and display |
| 20 | `themes` | `Friendship`, `First love`, `Family`, `College life`, `Coming of age`, `Underdog` and more | Generated from TMDB keywords | Filter and display |
| 21 | `intensity` | `Light`, `Medium`, `Heavy` | Generated by rules | Filter |
| 22 | `watch_with` | `Alone`, `Friends`, `Family`, `Date` | Generated by rules | Filter |
| 23 | `content_notes` | `Violence`, `Gore`, `Sexual content`, `Drugs`, `Suicide or self harm`, `Strong language`, `Adults only (A)` | TMDB keywords + certificate | Guardrail |
| 24 | `hook` | One line pitch: official tagline, else first plot sentence | TMDB | Display |
| 25 | `overview` | Official one paragraph plot | TMDB | Vibe and display |
| 26 | `director_or_creator` | Director (movie) or creator (series) | TMDB | Display and filter |
| 27 | `top_cast` | Top 3 billed cast | TMDB | Display and filter |
| 28 | `rating` | Average audience rating out of 10 | TMDB | Rank and filter |
| 29 | `vote_count` | How many people rated it | TMDB | Rank |
| 30 | `popularity` | TMDB popularity score | TMDB | Rank, hidden gems |
| 31 | `streaming_in` | Main platforms where it streams in India (subscription or free). Rent, buy and paid add on channels are in the `Where to watch` sheet. | JustWatch via TMDB | Filter and display |
| 32 | `free_to_watch` | `TRUE` if free or free with ads somewhere in India | JustWatch via TMDB | Filter |
| 33 | `watch_page_url` | Page listing every legal place to watch it | JustWatch via TMDB | Display |
| 34 | `poster_url` | Poster image link | TMDB | Display |
| 35 | `tmdb_url` | TMDB page (citation) | TMDB | Citation |
| 36 | `imdb_url` | IMDb page link | TMDB external IDs | Display |
| 37 | `has_plot` | `FALSE` when TMDB has no English plot yet | Derived | Quality check |
| 38 | `search_text` | Exact text that gets turned into an embedding | Built from the columns above | Embedding input |
| 39 | `data_as_of` | Date the row was fetched | Pipeline | Freshness |

### 9.2 `Where to watch` Sheet (7 Columns)

| # | Column | Meaning | Source | Used For |
|---|---|---|---|---|
| 1 | `title_id` | Links to `Titles` | Pipeline | Join key |
| 2 | `title` | Title for readability | TMDB | Display |
| 3 | `platform` | `Netflix`, `Amazon Prime Video`, `JioHotstar`, `ZEE5`, `SonyLIV`, `aha`, `Sun NXT`, `YouTube` and more | JustWatch via TMDB | Filter |
| 4 | `access_type` | `Subscription`, `Free`, `Free with ads`, `Rent`, `Buy` | JustWatch via TMDB | Filter |
| 5 | `region` | `IN` | Pipeline | Filter |
| 6 | `as_of` | Date availability was checked. It changes every month | Pipeline | Freshness guardrail |
| 7 | `platform_group` | Clean platform name for filters (`Netflix`, `Amazon Prime Video`, `JioHotstar`, `Prime Video add on` and so on) | Derived | Filter |

### 9.3 Functional Grouping of Columns in the RAG Pipeline

- **Vector Embedding Input:** `search_text` (indexed in ChromaDB using FastEmbed `BAAI/bge-small-en-v1.5`; filtered/checked with `has_plot`)
- **Primary Join Key:** `title_id` (joins `Titles` with `Where to watch`)
- **Structured Metadata Filters:** `type`, `language`, `industry`, `origin_country`, `release_year`, `era`, `runtime_min`, `length_bucket`, `seasons`, `episodes`, `binge_hours`, `status`, `certification_in`, `genres`, `mood`, `themes`, `intensity`, `watch_with`, `director_or_creator`, `top_cast`, `rating`, `streaming_in`, `free_to_watch`, `platform`, `platform_group`, `access_type`, `region`
- **Semantic Vibe & Similarity Matching:** `search_text`, `genres`, `keywords`, `mood`, `themes`, `overview`
- **Ranking & Hidden Gems:** `rating`, `vote_count`, `popularity`
- **Safety & Freshness Guardrails:** `certification_in` (empty = Unknown), `content_notes`, `as_of`, `data_as_of`, `has_plot`
- **Card Display & Citations:** `title`, `original_title`, `release_year`, `language`, `type`, `runtime_min`, `seasons`, `rating`, `hook`, `overview`, `director_or_creator`, `top_cast`, `streaming_in`, `watch_page_url`, `poster_url`, `tmdb_url`, `imdb_url`, `platform`, `access_type`, `as_of`
