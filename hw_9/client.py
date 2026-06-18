import os
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://fastapi:8000")

st.set_page_config(page_title="Bot Chat", layout="wide")
st.title("Bot Chat (LLM + Classifier)")
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


def call_get_message(text: str) -> str | None:
    try:
        r = requests.post(
            f"{API_BASE}/get_message",
            json={
                "dialog_id": dialog_id,
                "last_msg_text": text,
                "last_message_id": str(uuid.uuid4()),
            },
            timeout=60,
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
    user_prob = call_predict(user_input, st.session_state.turn % 2)
    st.session_state.messages.append({
        "role": "user",
        "text": user_input,
        "prob": user_prob,
    })
    st.session_state.turn += 1

    bot_text = call_get_message(user_input)
    if bot_text:
        bot_prob = call_predict(bot_text, st.session_state.turn % 2)
        st.session_state.messages.append({
            "role": "assistant",
            "text": bot_text,
            "prob": bot_prob,
        })
        st.session_state.turn += 1

    st.rerun()

with st.sidebar:
    st.subheader("Debug")
    probs = [m["prob"] for m in st.session_state.messages if m.get("prob") is not None]
    if probs:
        st.metric("Messages", len(probs))
        st.metric("Avg bot-prob", f"{sum(probs)/len(probs):.4f}")
    st.divider()
    if st.button("New Dialog"):
        st.session_state.dialog_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.turn = 0
        st.rerun()
