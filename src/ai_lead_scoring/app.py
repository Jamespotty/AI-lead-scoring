import copy
import json
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from ai_lead_scoring.config import (
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    MAX_WORKERS,
    OPENAI_MODEL,
    SCORE_BATCH_SIZE,
    TEXT_FIELD_LIMIT,
)
from ai_lead_scoring.pipeline import build_icp_system_prompt, run_scoring_pipeline
from ai_lead_scoring.utils import (
    add_log,
    checkpoint_exists,
    clear_checkpoint,
    dataset_fingerprint,
    init_session_state,
    list_all_checkpoints,
    load_checkpoint,
    parse_uploaded_file,
    save_jsonl,
)

st.set_page_config(
    page_title="AI Lead Scoring",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()


def render_sidebar():
    with st.sidebar:
        st.markdown("# 🎯 AI Lead Scoring")
        st.caption("Pre-scraped data → ICP scoring")
        st.divider()

        for page in [
            "Dashboard",
            "Upload Data",
            "ICP Configuration",
            "Lead Scoring",
            "Results & Export",
            "Settings",
        ]:
            btn_type = "primary" if st.session_state.current_page == page else "secondary"
            if st.button(page, use_container_width=True, type=btn_type, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()

        st.divider()
        scored = st.session_state.scored_leads
        st.metric("Uploaded", len(st.session_state.raw_leads))
        st.metric("Scored", len(scored))
        if scored:
            c1, c2, c3 = st.columns(3)
            c1.metric("🔥", sum(1 for l in scored if l.get("intent") == "high"))
            c2.metric("🟡", sum(1 for l in scored if l.get("intent") == "warm"))
            c3.metric("❄️", sum(1 for l in scored if l.get("intent") == "low"))

        st.divider()
        with st.expander("🔑 OpenAI Key", expanded=not st.session_state.openai_key):
            st.session_state.openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.openai_key,
                placeholder="sk-...",
            )


def page_dashboard():
    st.title("📊 Dashboard")

    scored = st.session_state.scored_leads
    high = sum(1 for l in scored if l.get("intent") == "high")
    warm = sum(1 for l in scored if l.get("intent") == "warm")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uploaded", len(st.session_state.raw_leads))
    c2.metric("Scored", len(scored))
    c3.metric("🔥 High Intent", high)
    c4.metric("🟡 Warm Intent", warm)

    if scored:
        st.divider()
        st.subheader("Intent Distribution")
        st.bar_chart(
            pd.DataFrame(
                {
                    "Intent": ["High", "Warm", "Low"],
                    "Count": [high, warm, len(scored) - high - warm],
                }
            ).set_index("Intent")
        )

        st.divider()
        st.subheader("Score Distribution")
        df = pd.DataFrame(scored)
        if "score" in df.columns:
            st.bar_chart(df["score"].value_counts().sort_index())

    st.divider()
    st.subheader("Recent Activity")
    if st.session_state.activity_log:
        st.dataframe(
            pd.DataFrame(st.session_state.activity_log),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No activity yet.")


def page_upload_data():
    st.title("📁 Upload Data")
    st.caption("Upload pre-scraped social media data (CSV, JSON, JSONL).")

    uploaded = st.file_uploader("Choose file", type=["csv", "json", "jsonl"])

    if not uploaded:
        return

    with st.spinner("Parsing..."):
        records, columns = parse_uploaded_file(uploaded)

    if not records:
        st.error("No records found.")
        return

    st.success(f"✅ **{len(records):,}** records · **{len(columns)}** columns")
    st.session_state.raw_leads = records
    st.session_state.upload_columns = columns
    add_log(f"Uploaded {uploaded.name} → {len(records):,} records")

    st.divider()
    st.subheader("Column Mapping")

    cols_with_blank = ["(none)"] + columns

    def best_guess(candidates, pool):
        for candidate in candidates:
            for i, col in enumerate(pool):
                if candidate.lower() in col.lower() or col.lower() in candidate.lower():
                    return i
        return 0

    c1, c2 = st.columns(2)
    with c1:
        col_title = st.selectbox(
            "Post Title",
            cols_with_blank,
            index=best_guess(["title", "subject", "headline"], cols_with_blank),
        )
        col_body = st.selectbox(
            "Post Body / Comment *",
            cols_with_blank,
            index=best_guess(
                ["body", "text", "content", "selftext", "comment"], cols_with_blank
            ),
        )
        col_permalink = st.selectbox(
            "Permalink / URL",
            cols_with_blank,
            index=best_guess(["permalink", "url", "link", "post_url"], cols_with_blank),
        )
    with c2:
        col_author = st.selectbox(
            "Username / Author",
            cols_with_blank,
            index=best_guess(
                ["author", "username", "user", "handle", "screen_name"], cols_with_blank
            ),
        )
        col_engagement = st.selectbox(
            "Engagement Score",
            cols_with_blank,
            index=best_guess(
                ["score", "upvotes", "likes", "views", "engagement"], cols_with_blank
            ),
        )

    def norm(v):
        return "" if v == "(none)" else v

    st.session_state.col_title = norm(col_title)
    st.session_state.col_body = norm(col_body)
    st.session_state.col_permalink = norm(col_permalink)
    st.session_state.col_author = norm(col_author)
    st.session_state.col_engagement = norm(col_engagement)

    if st.session_state.col_title or st.session_state.col_body:
        with st.expander("🔍 Preview: what the model sees (first 3 rows)"):
            for i, row in enumerate(records[:3], 1):
                title_val = (
                    str(row.get(st.session_state.col_title, ""))[:200]
                    if st.session_state.col_title
                    else ""
                )
                body_val = (
                    str(row.get(st.session_state.col_body, ""))[:TEXT_FIELD_LIMIT]
                    if st.session_state.col_body
                    else ""
                )
                combined = (
                    f"{title_val} | {body_val}"
                    if title_val and body_val
                    else (title_val or body_val)
                )
                author = (
                    str(row.get(st.session_state.col_author, ""))[:80]
                    if st.session_state.col_author
                    else "—"
                )
                link = (
                    str(row.get(st.session_state.col_permalink, ""))[:120]
                    if st.session_state.col_permalink
                    else "—"
                )
                st.markdown(f"**Row {i}**")
                st.write(f"📝 {combined[:300]}{'...' if len(combined) > 300 else ''}")
                st.write(f"👤 {author}  ·  🔗 {link}")
                st.divider()

    st.divider()
    st.subheader("Data Preview")
    st.dataframe(pd.DataFrame(records[:10]), use_container_width=True, hide_index=True)


def page_icp_configuration():
    st.title("🎯 ICP Configuration")
    st.caption("Define your product and ideal customer. Embedded in every scoring prompt.")

    st.subheader("Product")
    st.session_state.icp_product_description = st.text_area(
        "What does your product do?",
        value=st.session_state.icp_product_description,
        height=100,
        placeholder="e.g. A subscription management platform that helps SaaS companies reduce churn and automate billing.",
    )

    st.subheader("Ideal Customer")
    st.session_state.icp_target_customer = st.text_area(
        "Who is your ideal customer?",
        value=st.session_state.icp_target_customer,
        height=80,
        placeholder="e.g. Founders and finance leads at B2B SaaS companies with 10-200 employees.",
    )

    st.subheader("Pain Points We Solve")
    st.session_state.icp_pain_points = st.text_area(
        "What problems are your customers trying to solve?",
        value=st.session_state.icp_pain_points,
        height=80,
        placeholder="e.g. Churn management, dunning automation, failed payment recovery.",
    )

    st.subheader("Signal Keywords")
    st.session_state.icp_keywords = st.text_input(
        "Keywords that signal buying intent (comma-separated)",
        value=st.session_state.icp_keywords,
        placeholder="e.g. subscription management, churn, MRR, stripe alternative, looking for a tool",
    )

    st.divider()
    if st.button("✅ Save ICP", type="primary"):
        add_log("ICP configuration saved")
        st.success("Saved. Proceed to Lead Scoring.")

    with st.expander("🔍 Preview scoring prompt"):
        st.code(build_icp_system_prompt(), language="text")


def page_lead_scoring():
    st.title("⚡ Lead Scoring")
    st.caption(
        f"Batch: {SCORE_BATCH_SIZE}/call · {MAX_WORKERS} workers · checkpoint every {CHECKPOINT_EVERY} batches"
    )

    if not st.session_state.raw_leads:
        st.warning("No data loaded. Go to **Upload Data** first.")
        return
    if not st.session_state.icp_product_description:
        st.warning("No ICP defined. Go to **ICP Configuration** first for better results.")
    if not st.session_state.openai_key:
        st.error("OpenAI API key required. Add it in the sidebar.")
        return

    leads = st.session_state.raw_leads
    n = len(leads)
    batches = (n + SCORE_BATCH_SIZE - 1) // SCORE_BATCH_SIZE
    est_cost = (n * 0.5 * 0.15 / 1000) + (n * 0.04 * 0.60 / 1000)

    c1, c2, c3 = st.columns(3)
    c1.metric("Leads", f"{n:,}")
    c2.metric("Batches", f"{batches:,}")
    c3.metric("Est. Cost", f"~${est_cost:.2f}")

    st.divider()
    st.subheader("Column Mapping")
    cm1, cm2, cm3, cm4, cm5 = st.columns(5)
    cm1.info(f"**Title:** `{st.session_state.col_title or 'none'}`")
    cm2.info(f"**Body:** `{st.session_state.col_body or '⚠️ not set'}`")
    cm3.info(f"**Permalink:** `{st.session_state.col_permalink or 'none'}`")
    cm4.info(f"**Author:** `{st.session_state.col_author or 'none'}`")
    cm5.info(f"**Engagement:** `{st.session_state.col_engagement or 'none'}`")

    if not st.session_state.col_title and not st.session_state.col_body:
        st.error("Map at least Post Title or Post Body in Upload Data.")
        return

    st.divider()

    fingerprint = dataset_fingerprint(leads)
    has_checkpoint = checkpoint_exists(fingerprint)

    if has_checkpoint:
        prior_count = len(load_checkpoint(fingerprint))
        st.info(
            f"♻️ **Interrupted run detected** — {prior_count:,} / {n:,} scored ({prior_count/n*100:.0f}%). {n-prior_count:,} remain."
        )
        col_r, col_d = st.columns(2)
        if col_r.button(
            f"▶️ Resume from {prior_count:,}", type="primary", use_container_width=True
        ):
            st.session_state["_scoring_mode"] = "resume"
            st.rerun()
        if col_d.button("🗑️ Start fresh", use_container_width=True):
            clear_checkpoint(fingerprint)
            st.session_state.pop("_scoring_mode", None)
            add_log(f"Checkpoint discarded ({fingerprint})")
            st.rerun()
        st.divider()

    max_rows = st.number_input(
        "Max leads to score", min_value=10, max_value=n, value=n, step=100
    )

    if st.button(
        "▶️ Resume Scoring" if has_checkpoint else "🚀 Run Lead Scoring",
        type="primary",
        use_container_width=True,
    ):
        subset = copy.deepcopy(leads[: int(max_rows)])
        fp = dataset_fingerprint(subset) if int(max_rows) < n else fingerprint

        progress_bar = st.progress(0.0, text="Scoring...")
        start_ts = time.time()
        sc = st.columns(4)
        s_total, s_high, s_warm, s_low = (
            sc[0].empty(),
            sc[1].empty(),
            sc[2].empty(),
            sc[3].empty(),
        )

        def update_progress(fraction: float):
            elapsed = time.time() - start_ts
            eta = (elapsed / fraction * (1 - fraction)) if fraction > 0 else 0
            progress_bar.progress(
                fraction, text=f"Scoring... {fraction*100:.0f}% · ETA {eta:.0f}s"
            )
            cp = load_checkpoint(fp)
            h = sum(1 for v in cp.values() if v.get("intent") == "high")
            w = sum(1 for v in cp.values() if v.get("intent") == "warm")
            s_total.metric("Scored", f"{len(cp):,}")
            s_high.metric("🔥 High", h)
            s_warm.metric("🟡 Warm", w)
            s_low.metric("❄️ Low", len(cp) - h - w)

        try:
            scored, errors = run_scoring_pipeline(
                leads=subset,
                openai_key=st.session_state.openai_key,
                col_title=st.session_state.col_title,
                col_body=st.session_state.col_body,
                col_permalink=st.session_state.col_permalink,
                col_author=st.session_state.col_author,
                col_engagement=st.session_state.col_engagement,
                fingerprint=fp,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0, text="Done!")
            st.session_state.scored_leads = scored
            st.session_state.scoring_errors = errors
            st.session_state.pop("_scoring_mode", None)

            elapsed = time.time() - start_ts
            high = sum(1 for l in scored if l.get("intent") == "high")
            warm = sum(1 for l in scored if l.get("intent") == "warm")
            low = len(scored) - high - warm

            add_log(
                f"Scored {len(scored):,} in {elapsed:.0f}s — High:{high} Warm:{warm} Low:{low}"
            )
            path = f"scored_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
            save_jsonl(path, scored)
            st.session_state.jsonl_paths["scoring"] = path
            st.success(
                f"✅ {len(scored):,} leads in {elapsed:.0f}s — 🔥 {high} · 🟡 {warm} · ❄️ {low}"
            )

            if errors:
                with st.expander(f"⚠️ {len(errors)} batch errors"):
                    for e in errors:
                        st.caption(e)

        except Exception as e:
            st.error(
                f"Scoring interrupted: {e}\n\nProgress is checkpointed. Click Resume to continue."
            )
            add_log(f"Scoring interrupted ({fp}): {e}", status="Error")

    all_cps = list_all_checkpoints()
    if all_cps:
        with st.expander(f"🗂️ Checkpoints ({len(all_cps)})"):
            st.dataframe(pd.DataFrame(all_cps), use_container_width=True, hide_index=True)
            if st.button("🧹 Clear all checkpoints", use_container_width=True):
                for cp in all_cps:
                    try:
                        os.remove(os.path.join(CHECKPOINT_DIR, cp["file"]))
                    except Exception:
                        pass
                add_log("All checkpoints cleared")
                st.rerun()


def page_results_export():
    st.title("📋 Results & Export")

    if not st.session_state.scored_leads:
        st.info("No scored leads yet. Run Lead Scoring first.")
        return

    scored = st.session_state.scored_leads
    high = [l for l in scored if l.get("intent") == "high"]
    warm = [l for l in scored if l.get("intent") == "warm"]
    low = [l for l in scored if l.get("intent") == "low"]

    st.subheader("Filter by Intent")
    c1, c2, c3 = st.columns(3)
    show_high = c1.checkbox(f"🔥 High ({len(high):,})", value=True)
    show_warm = c2.checkbox(f"🟡 Warm ({len(warm):,})", value=True)
    show_low = c3.checkbox(f"❄️ Low ({len(low):,})", value=False)

    filtered = []
    if show_high:
        filtered.extend(high)
    if show_warm:
        filtered.extend(warm)
    if show_low:
        filtered.extend(low)

    st.caption(f"**{len(filtered):,}** leads")

    if not filtered:
        st.info("No leads match the current filter.")
        return

    st.divider()
    df = pd.DataFrame(filtered)
    permalink_col = st.session_state.col_permalink

    if permalink_col and permalink_col in df.columns:
        display_df = df.head(500).copy()
        display_df[permalink_col] = display_df[permalink_col].apply(
            lambda u: f'<a href="{u}" target="_blank">🔗 open</a>'
            if pd.notna(u) and str(u).startswith("http")
            else u
        )
        st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.dataframe(df.head(500), use_container_width=True, hide_index=True)

    if len(filtered) > 500:
        st.caption(f"Showing first 500 of {len(filtered):,}. Download for full data.")

    st.divider()
    st.subheader("Downloads")

    def to_csv(records):
        return pd.DataFrame(records).to_csv(index=False).encode("utf-8")

    def to_jsonl(records):
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode(
            "utf-8"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "⬇️ All (CSV)",
        data=to_csv(scored),
        file_name="all_scored.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "🔥 High (CSV)",
        data=to_csv(high),
        file_name="high_intent.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c3.download_button(
        "🟡 Warm (CSV)",
        data=to_csv(warm),
        file_name="warm_intent.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c4.download_button(
        "⬇️ All (JSONL)",
        data=to_jsonl(scored),
        file_name="all_scored.jsonl",
        mime="application/jsonl",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Lead Inspector")

    label_col = st.session_state.col_author or st.session_state.col_title or None
    if label_col and label_col in df.columns:
        labels = [
            f"[{l.get('intent','?').upper()}] {str(l.get(label_col,''))[:60]}"
            for l in filtered[:200]
        ]
    else:
        labels = [
            f"[{l.get('intent','?').upper()}] Lead {i}"
            for i, l in enumerate(filtered[:200])
        ]

    idx = st.selectbox(
        "Select lead", range(len(labels)), format_func=lambda i: labels[i]
    )
    lead = filtered[idx]

    col1, col2 = st.columns([1, 2])
    with col1:
        intent = lead.get("intent", "?")
        st.metric(
            "Intent",
            {"high": "🔥 HIGH", "warm": "🟡 WARM", "low": "❄️ LOW"}.get(
                intent, intent.upper()
            ),
        )
        st.metric("Score", f"{lead.get('score', '?')} / 100")
        st.write(f"**Reason:** {lead.get('score_reason', '')}")
        if permalink_col:
            pl = lead.get(permalink_col, "")
            if pl and str(pl).startswith("http"):
                st.markdown(f"[🔗 View original post]({pl})")
    with col2:
        st.json({k: v for k, v in lead.items()})


def page_settings():
    st.title("⚙️ Settings")

    st.subheader("API Key")
    st.session_state.openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_key,
        placeholder="sk-...",
    )

    st.divider()
    st.subheader("Scoring Parameters")
    st.info(
        f"Batch: **{SCORE_BATCH_SIZE}** · Workers: **{MAX_WORKERS}** · Model: **{OPENAI_MODEL}** · "
        f"Text limit: **{TEXT_FIELD_LIMIT}** chars · Checkpoint every **{CHECKPOINT_EVERY}** batches"
    )

    st.divider()
    st.subheader("Danger Zone")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Scored Leads", use_container_width=True):
            st.session_state.scored_leads = []
            st.session_state.scoring_errors = []
            add_log("Cleared scored leads")
            st.rerun()
    with c2:
        if st.button("🔄 Reset Everything", use_container_width=True):
            saved_key = st.session_state.openai_key
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.session_state.openai_key = saved_key
            add_log("Reset all data")
            st.rerun()


def main():
    render_sidebar()
    page_map = {
        "Dashboard": page_dashboard,
        "Upload Data": page_upload_data,
        "ICP Configuration": page_icp_configuration,
        "Lead Scoring": page_lead_scoring,
        "Results & Export": page_results_export,
        "Settings": page_settings,
    }
    page_map.get(st.session_state.current_page, page_dashboard)()


if __name__ == "__main__":
    main()
