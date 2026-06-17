import uuid

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Echo Chat + Classifier", layout="wide")
st.title("Echo Chat + Bot Classifier")
st.caption(f"Backend: {API_BASE}")

if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

dialog_id = st.session_state.dialog_id


def call_predict(text: str, participant_index: int) -> float | None:
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={
                "text": text,
                "dialog_id": dialog_id,
                "id": str(uuid.uuid4()),
                "participant_index": participant_index,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["is_bot_probability"]
    except Exception:
        return None


def call_echo(text: str) -> str | None:
    try:
        r = requests.post(
            f"{API_BASE}/get_message",
            json={
                "dialog_id": dialog_id,
                "last_msg_text": text,
                "last_message_id": str(uuid.uuid4()),
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["new_msg_text"]
    except Exception:
        return None


st.subheader("Chat")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        prob_str = ""
        if msg.get("prob") is not None:
            prob_str = f"  `[bot-prob: {msg['prob']:.3f}]`"
        st.markdown(f"{msg['text']}{prob_str}")

user_input = st.chat_input("Type a message...")

if user_input:
    user_prob = call_predict(user_input, participant_index=st.session_state.turn % 2)
    st.session_state.messages.append({
        "role": "user",
        "text": user_input,
        "prob": user_prob,
    })
    st.session_state.turn += 1

    echo_text = call_echo(user_input)
    if echo_text:
        echo_prob = call_predict(echo_text, participant_index=st.session_state.turn % 2)
        st.session_state.messages.append({
            "role": "assistant",
            "text": echo_text,
            "prob": echo_prob,
        })
        st.session_state.turn += 1

    st.rerun()

with st.sidebar:
    st.subheader("Metrics")

    probs = [m["prob"] for m in st.session_state.messages if m.get("prob") is not None]

    if probs:
        avg_prob = sum(probs) / len(probs)
        st.metric("Messages classified", len(probs))
        st.metric("Avg bot-probability", f"{avg_prob:.4f}")
        st.metric("Max", f"{max(probs):.4f}")
        st.metric("Min", f"{min(probs):.4f}")

        pairs = []
        for i in range(0, len(st.session_state.messages) - 1, 2):
            user_msg = st.session_state.messages[i]
            bot_msg = st.session_state.messages[i + 1]
            if (user_msg["role"] == "user" and bot_msg["role"] == "assistant"
                    and user_msg.get("prob") is not None and bot_msg.get("prob") is not None):
                pairs.append((user_msg, bot_msg))

        if pairs:
            diffs = [abs(u["prob"] - b["prob"]) for u, b in pairs]
            st.metric("Consistency (lower = better)", f"{sum(diffs)/len(diffs):.4f}")
            st.caption("Mean |user_prob − echo_prob| across turns. Since the bot echoes, probabilities should be similar.")

    st.divider()
    st.subheader("History")
    for i, msg in enumerate(st.session_state.messages):
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        prob_display = f"{msg['prob']:.3f}" if msg.get("prob") is not None else "—"
        st.text(f"{role_icon} [{prob_display}] {msg['text'][:60]}{'...' if len(msg['text']) > 60 else ''}")

    st.divider()
    if st.button("New Dialog"):
        st.session_state.dialog_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.turn = 0
        st.rerun()
