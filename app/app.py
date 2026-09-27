"""
Streamlit frontend for the Misinformation Detector.

Thin client: collects input, calls the FastAPI backend over HTTP, renders
the response. No model logic here. Three tabs:
  - Single Analysis : paste one article, see prediction + explanation
  - Batch Analysis  : upload a CSV, see aggregate stats + download results
  - Review Queue    : reviewer dashboard, confirm/dismiss/relabel items

Run with (from repo root, API must already be running on port 8000):
    streamlit run app/app.py
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Misinformation Detector", layout="wide")
st.title("📰 Misinformation Detector")

tab_single, tab_batch, tab_queue = st.tabs(["Single Analysis", "Batch Analysis", "Review Queue"])


def render_result(result):
    """Shared rendering for a single prediction result dict (used by both
    Single Analysis and, per-row, inside the Review Queue detail view)."""
    label = result["label"]
    confidence = result["confidence"]

    if label == "real":
        st.success(f"### ✅ Likely Real  ·  {confidence:.0%} confidence")
    elif label == "fake":
        st.error(f"### 🚩 Likely Misinformation  ·  {confidence:.0%} confidence")
    else:
        st.warning(f"### ❓ Uncertain  ·  {confidence:.0%} confidence")
        st.caption("The model's confidence was too low to call this either way — flagged for review, not treated as a verdict.")

    st.progress(result["prob_real"], text=f"Real: {result['prob_real']:.0%}  |  Fake: {result['prob_fake']:.0%}")

    st.subheader("Source & Author Credibility")
    cred_col1, cred_col2 = st.columns(2)
    with cred_col1:
        if result.get("source_known"):
            st.metric("Source credibility", f"{result['source_credibility']:.0%}")
        else:
            st.metric("Source credibility", "Unknown source")
            st.caption("This source wasn't in the training data — using a global average, not source-specific history.")
    with cred_col2:
        if result.get("author_known"):
            st.metric("Author known", "Yes")
        else:
            st.metric("Author known", "No")
            st.caption("This author wasn't in the training data.")

    if result.get("explanation"):
        st.subheader("Why this prediction?")

        main_signal = result.get("main_signal")
        main_direction = result.get("main_signal_direction")
        if main_signal:
            direction_label = {"real": "Real", "fake": "Fake", "neutral": "Neutral"}.get(main_direction, main_direction)
            signal_label = {
                "linguistic": "the linguistic patterns of the article",
                "semantic": "the semantic content of the article",
                "credibility": "source and author credibility information",
            }.get(main_signal, main_signal)
            st.info(f"**Main signal: {main_signal.upper()}** — {signal_label} is the strongest "
                    f"contributing factor, pushing this prediction toward **{direction_label}**.")

        strengths = result.get("signal_strengths")
        if strengths:
            st.caption("Signal strength (magnitude of influence, not direction):")
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Linguistic", f"{strengths['linguistic']:.2f}")
            s_col2.metric("Semantic", f"{strengths['semantic']:.2f}")
            s_col3.metric("Credibility", f"{strengths['credibility']:.2f}")

        st.caption("Positive values push toward Real, negative values push toward Fake. "
                   "These describe model behavior — they don't independently verify the claims in the text.")

        for item in result["explanation"]:
            feature = item["feature"]
            contribution = item["contribution"]
            direction = "→ Real" if contribution > 0 else ("→ Fake" if contribution < 0 else "→ Neutral")
            st.write(f"**{feature}**  `{contribution:+.4f}`  {direction}")
            normalized = max(-1.0, min(1.0, contribution * 5))
            st.progress((normalized + 1) / 2)

        if not result.get("source_known") and not result.get("author_known"):
            st.caption("ℹ️ Credibility note: neither the source nor the author was found in the "
                       "training credibility database — the global fallback ratio was used instead.")
        elif not result.get("source_known"):
            st.caption("ℹ️ Credibility note: the source wasn't found in the training credibility database.")
        elif not result.get("author_known"):
            st.caption("ℹ️ Credibility note: the author wasn't found in the training credibility database.")
    else:
        st.info("Explanation unavailable — the SHAP background dataset (train_features.parquet) "
                "wasn't found when the API started.")

    st.caption("**Important:** these explanations describe model behavior — they do not "
               "independently verify whether the article's claims are true.")
    st.divider()
    st.caption(f"Review priority: **{result['review_priority']}**")


# ============================================================
# TAB 1 — Single Analysis
# ============================================================
with tab_single:
    st.caption("Paste an article or post below. This tool flags likely misinformation for human review — "
               "it does not independently verify facts.")

    with st.form("predict_form"):
        text = st.text_area("Article / post text", height=200, placeholder="Paste the full text here...")
        col1, col2 = st.columns(2)
        with col1:
            source = st.text_input("Source / publisher (optional)", placeholder="e.g. reuters.com")
        with col2:
            author = st.text_input("Author (optional)", placeholder="e.g. Jane Doe")
        submitted = st.form_submit_button("Analyze", use_container_width=True)

    if submitted:
        if not text.strip():
            st.warning("Enter some text first.")
            st.stop()
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json={"text": text, "source": source, "author": author}, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Can't reach the API at {API_URL} — is `uvicorn main:app` running (from the `api/` folder)?")
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()
        render_result(result)


# ============================================================
# TAB 2 — Batch Analysis
# ============================================================
with tab_batch:
    st.caption("Upload a CSV with at least a `text` column (optional `title`, `source`, `author`). "
               "Every row is run through the model; results are stored for review below.")

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded and st.button("Run batch analysis", use_container_width=True):
        with st.spinner("Processing batch — this can take a while for large files..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                resp = requests.post(f"{API_URL}/predict/batch/csv", files=files, timeout=300)
                resp.raise_for_status()
                summary = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Can't reach the API at {API_URL}.")
                st.stop()
            except Exception as e:
                st.error(f"Batch processing failed: {e}")
                st.stop()

        st.success(f"Processed {summary['total_items']} items.")
        st.session_state["last_batch_id"] = summary["batch_id"]

        counts = summary["label_counts"]
        cols = st.columns(len(counts) or 1)
        for col, (label, count) in zip(cols, counts.items()):
            col.metric(label.capitalize(), count)

        csv_url = f"{API_URL}/batches/{summary['batch_id']}/export.csv"
        try:
            csv_resp = requests.get(csv_url, timeout=30)
            st.download_button(
                "⬇ Download results as CSV",
                data=csv_resp.content,
                file_name=f"batch_{summary['batch_id']}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception:
            st.caption(f"Download manually from: {csv_url}")

        st.caption("These items now also appear in the **Review Queue** tab.")


# ============================================================
# TAB 3 — Review Queue (human-in-the-loop), grouped by batch
# ============================================================
with tab_queue:
    if "selected_group" not in st.session_state:
        st.session_state["selected_group"] = None  # None = show the group list

    # ---- Group list view ----
    if st.session_state["selected_group"] is None:
        st.caption("Every CSV batch (and single-analysis submissions) forms its own group. "
                   "Open a group to review its items.")

        if st.button("Refresh groups"):
            st.rerun()

        try:
            resp = requests.get(f"{API_URL}/batches", timeout=30)
            resp.raise_for_status()
            groups = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach the API at {API_URL}.")
            groups = []
        except Exception as e:
            st.error(f"Couldn't load groups: {e}")
            groups = []

        if not groups:
            st.info("No analyzed content yet — run something in Single Analysis or Batch Analysis first.")

        for g in groups:
            with st.container(border=True):
                name = g["filename"] or "(unnamed batch)"
                when = f" · {g['submitted_at'][:19]}" if g.get("submitted_at") else ""
                st.write(f"**{name}**{when}")

                counts_col, action_col = st.columns([4, 1])
                with counts_col:
                    label_counts = g["label_counts"]
                    status_counts = g["status_counts"]
                    label_str = " · ".join(f"{k}: {v}" for k, v in label_counts.items())
                    pending = status_counts.get("pending", 0)
                    st.caption(f"{g['total_items']} item(s)  ·  {label_str}  ·  {pending} pending review")
                with action_col:
                    if st.button("Open →", key=f"open_{g['id'] or 'none'}", use_container_width=True):
                        st.session_state["selected_group"] = g["id"] if g["id"] else "__NONE__"
                        st.session_state["selected_group_name"] = name
                        st.rerun()

    # ---- Drill-in view: items within one selected group ----
    else:
        selected = st.session_state["selected_group"]
        group_name = st.session_state.get("selected_group_name", "")

        back_col, title_col = st.columns([1, 5])
        with back_col:
            if st.button("← Back to groups"):
                st.session_state["selected_group"] = None
                st.rerun()
        with title_col:
            st.write(f"**Reviewing: {group_name}**")

        st.caption("Sorted by confidence ascending by default — the items the model is LEAST sure "
                   "about surface first, since those need review most.")

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            f_status = st.selectbox("Review status", ["(any)", "pending", "confirmed", "dismissed", "relabeled"])
        with filter_col2:
            f_label = st.selectbox("Predicted label", ["(any)", "real", "fake", "uncertain"])
        with filter_col3:
            f_source = st.text_input("Source contains", placeholder="e.g. blog")
        with filter_col4:
            f_sort = st.selectbox("Sort by", ["confidence", "created_at", "label", "source"])

        params = {"sort_by": f_sort, "sort_dir": "asc", "limit": 200}
        if selected == "__NONE__":
            params["no_batch"] = True
        else:
            params["batch_id"] = selected
        if f_status != "(any)":
            params["review_status"] = f_status
        if f_label != "(any)":
            params["label"] = f_label
        if f_source:
            params["source"] = f_source

        if st.button("Refresh items"):
            st.rerun()

        try:
            resp = requests.get(f"{API_URL}/queue", params=params, timeout=30)
            resp.raise_for_status()
            items = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach the API at {API_URL}.")
            items = []
        except Exception as e:
            st.error(f"Couldn't load queue: {e}")
            items = []

        st.caption(f"{len(items)} item(s) in this group")

        for item in items:
            status_emoji = {"pending": "🕒", "confirmed": "✅", "dismissed": "🚫", "relabeled": "✏️"}.get(item["review_status"], "")
            label_emoji = {"real": "✅", "fake": "🚩", "uncertain": "❓"}.get(item["label"], "")
            header = f"{status_emoji} {label_emoji} **{item['label'].upper()}** ({item['confidence']:.0%}) — {item['text'][:80]}..."

            with st.expander(header):
                st.write(f"**Source:** {item['source'] or '(none)'}  |  **Author:** {item['author'] or '(none)'}")
                st.write(f"**Full text:** {item['text']}")
                st.write(f"**Review status:** `{item['review_status']}`  |  **Review priority:** `{item['review_priority']}`")

                detail_resp = requests.get(f"{API_URL}/predictions/{item['id']}", timeout=10)
                history = detail_resp.json().get("feedback_history", []) if detail_resp.ok else []
                if history:
                    st.write("**Feedback history:**")
                    for h in history:
                        label_note = f" → relabeled as `{h['corrected_label']}`" if h.get("corrected_label") else ""
                        st.caption(f"- `{h['action']}`{label_note} {('— ' + h['note']) if h.get('note') else ''} ({h['created_at']})")

                fb_col1, fb_col2, fb_col3 = st.columns(3)
                with fb_col1:
                    if st.button("✅ Confirm", key=f"confirm_{item['id']}", use_container_width=True):
                        requests.post(f"{API_URL}/predictions/{item['id']}/feedback", json={"action": "confirm"})
                        st.rerun()
                with fb_col2:
                    if st.button("🚫 Dismiss", key=f"dismiss_{item['id']}", use_container_width=True):
                        requests.post(f"{API_URL}/predictions/{item['id']}/feedback", json={"action": "dismiss"})
                        st.rerun()
                with fb_col3:
                    relabel_target = "real" if item["label"] != "real" else "fake"
                    if st.button(f"✏️ Relabel as {relabel_target}", key=f"relabel_{item['id']}", use_container_width=True):
                        requests.post(f"{API_URL}/predictions/{item['id']}/feedback",
                                      json={"action": "relabel", "corrected_label": relabel_target})
                        st.rerun()
