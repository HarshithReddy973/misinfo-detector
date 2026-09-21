# Datasets — status and decisions (updated after seeing actual raw files)

## FakeNewsNet-raw (gossipcop_fake/real.csv, politifact_fake/real.csv)
Columns: `id, news_url, title, tweet_ids` — **no article body text**.
**Decision: not used for training.** No `text` column, scraping the URLs
is too slow/unreliable for 5 days. Keeping the files in case there's time
later to derive an engagement (`share_count`) feature by matching IDs
against the main FakeNewsNet dataset — not a Day-1/2 priority.

## FakeNewsNet (main, richer folder) — RESOLVED
Confirmed as the mdepak/fakenewsnet Kaggle mirror (BuzzFeed + PolitiFact).
Label comes from filename, not a column — separate `*_fake_news_content.csv`
/ `*_real_news_content.csv` per outlet. Adapter: `clean_fakenewsnet_main.py`.

**Size note:** this is the original small Shu et al. dataset — a few
hundred rows total across BuzzFeed + PolitiFact combined, not thousands.
It contributes real `source`/`author`/`publish_date` metadata quality,
not training volume — ISOT and LIAR carry the volume. Say this explicitly
in the report so it reads as a scoping decision, not a gap.

**Skipped on purpose:** the `*NewsUser.txt`, `*UserUser.txt`,
`*UserFeature.mat` files are the social propagation graph (who shared
what, follower network). Real signal for the PS's optional propagation
extension, but out of scope for a 5-day build. Mention as a stretch goal
you scoped out, not something missed.

## ISOT
`title, text, subject, date` — `subject` is a topic tag (politics/world
news), not a publisher, so it is **not** mapped to `source`. No author,
no real source, no engagement data.

## LIAR
6-way label collapsed to a **3-way** real/uncertain/fake (not forced
binary) — see `clean_liar.py`. This lines up with the product's existing
"uncertain" label and is a deliberate design choice worth stating in the
midterm report, not just a cleanup detail.

---

## FakeNewsNet (main) — CONFIRMED: Kaggle mdepak/fakenewsnet mirror
4 CSVs, label from filename: `BuzzFeed_fake/real_news_content.csv`,
`PolitiFact_fake/real_news_content.csv`. Columns match: id, title, text,
url, top_img, authors, source, publish_date, movies, images,
canonical_link, meta_data. `authors` is a stringified list, parsed by the
adapter. BuzzFeed subset is small (~90/90) — PolitiFact carries most of
the volume. Social-graph files (*NewsUser.txt, *UserUser.txt, *.mat)
deliberately skipped — propagation graphs are an optional stretch goal
per the PS, not core.

## Pipeline order
1. `clean_fakenewsnet_main.py` → `data/processed/fakenewsnet_main_clean.parquet`
2. `clean_isot.py` → `data/processed/isot_clean.parquet`
3. `clean_liar.py` → `data/processed/liar_clean.parquet`
4. `merge_all.py` → `data/processed/combined_clean.parquet`  ← Person B splits this

---

# FakeNewsNet — what's actually available

Share this with the team once cleaning is done — it determines what the
credibility scorer and dashboard can honestly promise.

## Confirmed available (Kaggle CSV mirror)
- `title`, `text` (article body), `source_domain`, `real` (label)
- `tweet_num` → mapped to `share_count` (this IS real propagation signal)

## NOT available (known gaps — don't build features assuming these exist)
- `author` — not reliably present. Credibility scoring must be **source-level**
  (by domain), not author-level, unless we find a richer mirror.
- `timestamp` — not in the standard mirror. No time-based features unless we
  find a version that has it.
- `reply_count` — not available. Only share/tweet count.

## Label direction (double-check this against whatever CSV you actually load)
`real` column: 1 = real, 0 = fake, in the mirrors checked so far.
**Verify this on your specific download** — flipped labels are a classic
last-minute bug that silently tanks your accuracy number.

## Recommendation for Person B (train/test split)
Do the split by `source` (domain), not randomly — several sources appear
many times with a fixed label, so a random split leaks source identity into
train and test and inflates your score. A source-held-out or topic-held-out
split is more honest and defensible in the demo.
