"""
Streamlit frontend for the Misinformation Detector.

This is a THIN CLIENT: all it does is collect input, call the FastAPI
backend's /predict endpoint over HTTP, and render the response. No model
logic lives here -- that's the API/model track's job. This separation is
why the two run as independent processes (see README's "Streamlit vs
FastAPI" section).

Run with (from repo root, API must already be running on port 8000):
    streamlit run app/app.py
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Misinformation Detector", layout="centered")
st.title("📰 Misinformation Detector")
st.caption("Paste an article or post below. This tool flags likely misinformation for human review — "
           "it does not independently verify facts.")

with st.form("predict_form"):
    text = st.text_area(
        "Article / post text",
        height=200,
        placeholder="Paste the full text here...",
    )
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
            resp = requests.post(
                f"{API_URL}/predict",
                json={"text": text, "source": source, "author": author},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach the API at {API_URL} — is `uvicorn main:app` running (from the `api/` folder)?")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ---- Result banner ----
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

    # ---- Credibility ----
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

    # ---- Explanation ----
    if result["explanation"]:
        st.subheader("Why this prediction?")

        # "Main signal" summary sentence — which block (linguistic/semantic/
        # credibility) actually drove this prediction, matching predict_raw.py's CLI output
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

        # Signal strength comparison — which block carries the most weight overall
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

        # Credibility fallback note, matching predict_raw.py's CLI output
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

    # ---- Review priority (feeds the future reviewer dashboard) ----
    st.divider()
    st.caption(f"Review priority: **{result['review_priority']}**")
