import streamlit as st
import numpy as np
import pickle
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess
from model import build_model
from decision_enginee import get_recommendation

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Emotion AI Assistant",
    page_icon="🧠",
    layout="centered"
)

# ------------------ TITLE ------------------
st.markdown("""
    <h1 style='text-align: center; color: #6C63FF;'>
        🧠 Emotion-Aware Recommendation System
    </h1>
""", unsafe_allow_html=True)

st.markdown("### 💬 Understand your emotions & get smart suggestions")

# ------------------ INPUT ------------------
user_input = st.text_area(
    "✍️ Enter your thoughts:",
    placeholder="Example: I feel very stressed and tired today..."
)

# ------------------ LOAD MODEL & TOKENIZER ------------------
@st.cache_resource
def load_model_and_tokenizer():
    
    # Load tokenizer
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    with open("tokenizer.json") as f:
         data = f.read()

         tokenizer = tokenizer_from_json(data)

    # Build model (same architecture as training)
    from tensorflow.keras.models import load_model

    model = load_model("model.h5")

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ------------------ PREDICTION BUTTON ------------------
if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")

    else:
        # -------- PREPROCESS --------
        temp_df = pd.DataFrame({"journal_text": [user_input]})
        temp_df, _, _ = preprocess(temp_df, is_train=False)

        # -------- TOKENIZE --------
        seq = tokenizer.texts_to_sequences(temp_df['journal_text'])
        padded = pad_sequences(seq, maxlen=30, padding='post')

        # -------- PREDICT --------
        # -------- TOKENIZE --------
        seq = tokenizer.texts_to_sequences(temp_df['journal_text'])
        padded = pad_sequences(seq, maxlen=30, padding='post')

        # -------- PREDICT --------
        preds = model.predict(padded)

        # ✅ DEFINE pred_label HERE
        pred_label = np.argmax(preds, axis=1)[0]

        # -------- LOAD ENCODER --------
        with open("label_encoder.pkl", "rb") as f:
            target_encoder = pickle.load(f)

        # ✅ NOW use pred_label
        emotion = target_encoder.inverse_transform([pred_label])[0]
        # -------- LABEL MAPPING --------
        

        # Temporary intensity (future improvement: predict this)
        intensity = 5

        # -------- DECISION ENGINE --------
        _, _, action, timing = get_recommendation(emotion, intensity)

        # ------------------ OUTPUT ------------------
        st.markdown("---")

        st.success(f"🧠 Emotion: **{emotion.upper()}**")
        st.info(f"🔥 Intensity: **{intensity}/10**")

        st.markdown("### 🎯 Recommendation")
        st.write(f"👉 **Action:** {action}")
        st.write(f"⏰ **Best Time:** {timing}")

        # Fun UI touch
        if emotion == "happy":
            st.balloons()