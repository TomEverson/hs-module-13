import os
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://orchestrator:8000")

st.set_page_config(
    page_title="TomBot — Turing Arena",
    page_icon="🤖",
    layout="wide",
)

# ── session state ─────────────────────────────────────────────────────────────
if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn" not in st.session_state:
    st.session_state.turn = 0


# ── api helpers ───────────────────────────────────────────────────────────────

def call_predict(text: str, participant_index: int) -> float | None:
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={
                "text": text,
                "dialog_id": st.session_state.dialog_id,
                "id": str(uuid.uuid4()),
                "participant_index": participant_index,
            },
            timeout=15,
        )
        r.raise_for_status()
        return round(r.json()["is_bot_probability"], 4)
    except Exception:
        return None


def call_get_message(text: str) -> str | None:
    try:
        r = requests.post(
            f"{API_BASE}/get_message",
            json={
                "dialog_id": st.session_state.dialog_id,
                "last_msg_text": text,
                "last_message_id": str(uuid.uuid4()),
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["new_msg_text"]
    except Exception:
        return None


# ── ui helpers ────────────────────────────────────────────────────────────────

def prob_badge(prob: float | None) -> str:
    if prob is None:
        return ""
    color = "#ff4444" if prob >= 0.6 else "#ffaa00" if prob >= 0.4 else "#22cc66"
    label = "BOT" if prob >= 0.6 else "?" if prob >= 0.4 else "HUMAN"
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:0.75em;font-weight:bold;margin-left:8px">'
        f'{label} {prob:.3f}</span>'
    )


def gauge_html(prob: float) -> str:
    fill = int(prob * 100)
    color = "#ff4444" if prob >= 0.6 else "#ffaa00" if prob >= 0.4 else "#22cc66"
    return f"""
    <div style="margin:8px 0">
      <div style="display:flex;justify-content:space-between;
                  font-size:0.8em;color:#888;margin-bottom:4px">
        <span>HUMAN</span><span>BOT</span>
      </div>
      <div style="background:#333;border-radius:8px;height:18px;overflow:hidden">
        <div style="background:{color};width:{fill}%;height:100%;
                    border-radius:8px;transition:width 0.4s ease"></div>
      </div>
      <div style="text-align:center;font-size:1.6em;font-weight:bold;
                  color:{color};margin-top:6px">{prob:.1%}</div>
    </div>
    """


# ── layout ────────────────────────────────────────────────────────────────────

st.markdown("## 🤖 TomBot — Turing Arena")
st.caption("The LLM tries to pass as human. The classifier tries to catch it.")

col_chat, col_stats = st.columns([3, 1])

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(
                msg["text"] + prob_badge(msg.get("prob")),
                unsafe_allow_html=True,
            )

    user_input = st.chat_input("Say something…")

    if user_input:
        user_prob = call_predict(user_input, st.session_state.turn % 2)
        st.session_state.messages.append(
            {"role": "user", "text": user_input, "prob": user_prob}
        )
        st.session_state.turn += 1

        bot_text = call_get_message(user_input)
        if bot_text:
            bot_prob = call_predict(bot_text, st.session_state.turn % 2)
            st.session_state.messages.append(
                {"role": "assistant", "text": bot_text, "prob": bot_prob}
            )
            st.session_state.turn += 1

        st.rerun()

with col_stats:
    st.markdown("### Bot-O-Meter")

    probs = [m["prob"] for m in st.session_state.messages if m.get("prob") is not None]
    last_prob = probs[-1] if probs else 0.5

    st.markdown(gauge_html(last_prob), unsafe_allow_html=True)

    if probs:
        avg = sum(probs) / len(probs)
        verdict = "🔴 BOT" if avg >= 0.6 else "🟡 UNCERTAIN" if avg >= 0.4 else "🟢 HUMAN"
        st.metric("Session avg", f"{avg:.3f}")
        st.markdown(f"**Verdict: {verdict}**")
        st.metric("Messages scored", len(probs))
    else:
        st.caption("Start chatting to see scores")

    st.divider()

    if st.button("New Round", use_container_width=True):
        st.session_state.dialog_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.turn = 0
        st.rerun()

    st.divider()
    st.caption(f"API: `{API_BASE}`")
    st.caption(f"Dialog: `{st.session_state.dialog_id[:8]}…`")
