import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import streamlit as st
from openai import OpenAI

from ai_lead_scoring.config import (
    CHECKPOINT_EVERY,
    MAX_WORKERS,
    OPENAI_MODEL,
    SCORE_BATCH_SIZE,
    TEXT_FIELD_LIMIT,
)
from ai_lead_scoring.utils import (
    _extract_json_array,
    append_checkpoint,
    clear_checkpoint,
    load_checkpoint,
)


def build_icp_system_prompt() -> str:
    p = st.session_state.icp_product_description.strip()
    c = st.session_state.icp_target_customer.strip()
    pp = st.session_state.icp_pain_points.strip()
    kw = st.session_state.icp_keywords.strip()

    return f"""You are an expert B2B/B2C lead qualifier. Classify social media posts/comments as leads for a specific product.

=== PRODUCT / ICP CONTEXT ===
Product: {p or "Not specified"}
Ideal Customer Profile: {c or "Not specified"}
Pain Points We Solve: {pp or "Not specified"}
Signal Keywords: {kw or "Not specified"}

=== INTENT CLASSIFICATION RULES ===
HIGH: Author clearly expresses a problem our product solves, is actively seeking a solution, mentions budget/tools/switching, or is in decision-making mode.
WARM: Author shows indirect interest — discussing the problem space, asking for recommendations, or mild awareness.
LOW:  Not relevant to our product or ICP. No signal, spam, off-topic, or competitor content.

=== OUTPUT RULES ===
- Return ONLY a valid JSON array, no markdown, no extra text.
- Score 0-100 (80-100 = high, 50-79 = warm, 0-49 = low).
- Keep "reason" under 20 words.
- Schema: {{"id": "...", "intent": "high|warm|low", "score": 0-100, "reason": "..."}}
"""


def score_batch(
    batch: List[Dict[str, Any]],
    openai_key: str,
    col_title: str,
    col_body: str,
    col_author: str,
    col_engagement: str,
    system_prompt: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    client = OpenAI(api_key=openai_key)
    payload = []
    for lead in batch:
        title = str(lead.get(col_title, "")).strip()[:200] if col_title else ""
        body = str(lead.get(col_body, "")).strip()[:TEXT_FIELD_LIMIT] if col_body else ""
        combined = f"{title} | {body}" if title and body else (title or body)
        item = {"id": lead["_lead_id"], "text": combined}
        if col_author and lead.get(col_author):
            item["author"] = str(lead[col_author])[:80]
        if col_engagement and lead.get(col_engagement):
            item["engagement"] = str(lead[col_engagement])[:40]
        payload.append(item)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Classify each lead. Return a JSON array.\n\n"
                    + json.dumps(payload, ensure_ascii=False),
                },
            ],
            temperature=0,
            max_tokens=1500,
        )
        msg_content = resp.choices[0].message.content
        if msg_content is None:
            raise ValueError("No content returned from OpenAI")
        return _extract_json_array(msg_content), []
    except Exception as e:
        err = f"Batch error (ids {batch[0]['_lead_id']}..{batch[-1]['_lead_id']}): {e}"
        fallback = [
            {"id": lead["_lead_id"], "intent": "low", "score": 0, "reason": "Scoring error"}
            for lead in batch
        ]
        return fallback, [err]


def run_scoring_pipeline(
    leads: List[Dict[str, Any]],
    openai_key: str,
    col_title: str,
    col_body: str,
    col_permalink: str,
    col_author: str,
    col_engagement: str,
    fingerprint: str,
    progress_callback=None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    system_prompt = build_icp_system_prompt()

    for i, lead in enumerate(leads):
        lead["_lead_id"] = str(i)

    score_map: Dict[str, Dict] = load_checkpoint(fingerprint)
    already_done = len(score_map)

    all_batches = [
        leads[i : i + SCORE_BATCH_SIZE] for i in range(0, len(leads), SCORE_BATCH_SIZE)
    ]
    pending = [
        b for b in all_batches if not all(lead["_lead_id"] in score_map for lead in b)
    ]

    all_errors: List[str] = []
    completed = already_done // SCORE_BATCH_SIZE
    unflushed: List[Dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                score_batch,
                batch,
                openai_key,
                col_title,
                col_body,
                col_author,
                col_engagement,
                system_prompt,
            ): batch
            for batch in pending
        }
        for future in as_completed(futures):
            results, errors = future.result()
            all_errors.extend(errors)
            for r in results:
                score_map[str(r.get("id", ""))] = r
                unflushed.append(r)
            completed += 1
            if len(unflushed) >= CHECKPOINT_EVERY * SCORE_BATCH_SIZE:
                append_checkpoint(fingerprint, unflushed)
                unflushed.clear()
            if progress_callback:
                progress_callback(min(completed / len(all_batches), 1.0))

    if unflushed:
        append_checkpoint(fingerprint, unflushed)

    scored = []
    for lead in leads:
        sr = score_map.get(lead["_lead_id"], {})
        merged = {
            "intent": sr.get("intent", "low"),
            "score": sr.get("score", 0),
            "score_reason": sr.get("reason", ""),
        }
        for col in [col_author, col_title, col_body, col_permalink, col_engagement]:
            if col and col in lead:
                merged[col] = lead[col]
        scored.append(merged)

    if len(score_map) >= len(leads):
        clear_checkpoint(fingerprint)

    intent_order = {"high": 0, "warm": 1, "low": 2}
    scored.sort(
        key=lambda x: (intent_order.get(x.get("intent", "low"), 2), -x.get("score", 0))
    )
    return scored, all_errors
