import streamlit as st
import joblib
import re
import html

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MindCare AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Page background */
    .stApp {
        background-color: #f6f8fc;
    }

    /* Main content width */
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Titles */
    h1 {
        color: #18233a;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: #24324a;
    }

    /* Text area */
    textarea {
        border-radius: 14px !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        height: 48px;
    }

    /* Divider */
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Small footer */
    .footer-text {
        text-align: center;
        color: #7b879d;
        font-size: 0.8rem;
        margin-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LABEL MAP
# ============================================================

label_map = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar Disorder",
    3: "Personality Disorder",
    4: "Anxiety"
}


# ============================================================
# LOAD MODEL + NOMIC
# ============================================================

@st.cache_resource
def load_models():

    artifacts = joblib.load(
        "mental_health_svm_artifacts.pkl"
    )

    svm_model = artifacts["model"]

    embedding_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )

    return svm_model, embedding_model


with st.spinner("Loading AI models..."):
    svm_model, embedding_model = load_models()


# ============================================================
# GROQ CLIENT
# ============================================================

groq_api_key = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=groq_api_key
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = html.unescape(text)

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# HIGH-RISK DETECTION
# ============================================================

HIGH_RISK_KEYWORDS = [
    "suicide",
    "suicidal",
    "kill myself",
    "killing myself",
    "end my life",
    "take my life",
    "want to die",
    "wish i was dead",
    "wish i were dead",
    "better off dead",
    "don't want to live",
    "dont want to live",
    "no reason to live",
    "self harm",
    "self-harm",
    "selfharm",
    "hurt myself",
    "harm myself",
    "cut myself",
    "overdose",
    "commit suicide"
]


def detect_high_risk(text):

    text_lower = str(text).lower()

    for keyword in HIGH_RISK_KEYWORDS:

        if keyword in text_lower:
            return True

    return False


# ============================================================
# ML CLASSIFICATION
# ============================================================

def predict_category(text):

    cleaned_text = clean_text(text)

    # Same prefix used during training
    nomic_text = "classification: " + cleaned_text

    embedding = embedding_model.encode(
        [nomic_text],
        normalize_embeddings=True
    )

    prediction = svm_model.predict(
        embedding
    )[0]

    return label_map[int(prediction)]


# ============================================================
# LLM RESPONSE GENERATION
# ============================================================

def generate_response(
    text,
    category,
    high_risk
):

    system_prompt = """
You are the response-generation component of a mental-health NLP project.

A machine-learning classifier has already classified the text into one
of these categories:

0 = Stress
1 = Depression
2 = Bipolar Disorder
3 = Personality Disorder
4 = Anxiety

Your task is to generate a safe, concise and supportive response based
on the original text and classifier prediction.

IMPORTANT RULES:

1. The classifier prediction is NOT a medical diagnosis.
2. Never say or imply that the person definitely has the condition.
3. Do not invent symptoms, feelings, experiences, or circumstances.
4. Do not change the classifier's predicted category.
5. Always use the FULL category name.
6. Give EXACTLY 3 practical supportive suggestions.
7. Do not prescribe medication or specific medical treatment.
8. Use empathetic, neutral and non-judgmental language.
9. ALWAYS provide a Safety Note.
10. Do not mention country-specific emergency numbers.
11. If high-risk is True, make the Safety Note more urgent.
12. If the text is informational, an announcement, or a community post,
    do not pretend that the author personally has the condition.
13. Base the explanation only on the actual text.
14. Keep the response concise.

Return EXACTLY:

Detected Category:
[full category name]

Brief Explanation:
[1-2 concise sentences]

Supportive Suggestions:
1. ...
2. ...
3. ...

When to Seek Professional Help:
[1-2 concise sentences]

Safety Note:
[1 concise safety statement]
"""

    user_prompt = f"""
Original Reddit text:
{text}

Classifier prediction:
{category}

High-risk indicator:
{high_risk}

Generate the response using the required format.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=800
    )

    return response.choices[0].message.content


# ============================================================
# HEADER
# ============================================================

st.title("🧠 MindCare AI")

st.subheader(
    "Mental Health Text Classification & Support"
)

st.write(
    "Analyze a Reddit-style mental-health post using "
    "Nomic embeddings and a Linear SVM classifier, then "
    "generate a supportive response using an LLM."
)

st.caption(
    "Nomic Embeddings  •  Linear SVM  •  High-Risk Detection  •  Groq LLM"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔄 System Pipeline")

    st.write("1️⃣ User enters text")
    st.write("2️⃣ Text cleaning")
    st.write("3️⃣ Nomic embedding")
    st.write("4️⃣ Linear SVM classification")
    st.write("5️⃣ High-risk keyword check")
    st.write("6️⃣ Groq GPT-OSS-20B")
    st.write("7️⃣ Supportive response")

    st.divider()

    st.header("📋 Categories")

    for class_id, category in label_map.items():

        st.write(
            f"**{class_id}** — {category}"
        )

    st.divider()

    st.warning(
        "This application is for educational and research purposes. "
        "It does not provide a medical diagnosis."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("✍️ Analyze a Post")

st.write(
    "Paste a Reddit post or enter mental-health-related text below."
)

text = st.text_area(
    "Input text",
    height=230,
    placeholder=(
        "Example:\n\n"
        "I have been feeling overwhelmed by college deadlines "
        "and exams. I keep worrying about everything and find "
        "it difficult to relax."
    ),
    label_visibility="collapsed"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_clicked = st.button(
    "🔍 Analyze Text",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS
# ============================================================

if analyze_clicked:

    if not text.strip():

        st.warning(
            "Please enter some text before clicking Analyze Text."
        )

    else:

        try:

            with st.spinner(
                "Running classification and generating supportive guidance..."
            ):

                # ML
                category = predict_category(text)

                # Safety check
                high_risk = detect_high_risk(text)

                # LLM
                llm_response = generate_response(
                    text=text,
                    category=category,
                    high_risk=high_risk
                )

            # ====================================================
            # RESULT
            # ====================================================

            st.divider()

            st.header("📊 Analysis Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    label="Detected Category",
                    value=category
                )

            with col2:

                if high_risk:

                    st.error(
                        "⚠️ High-risk language detected"
                    )

                else:

                    st.success(
                        "✓ No predefined high-risk keywords detected"
                    )


            # ====================================================
            # SUPPORTIVE RESPONSE
            # ====================================================

            st.subheader("💬 AI Supportive Response")

            st.markdown(
                llm_response
            )

        except Exception as e:

            st.error(
                f"An error occurred while processing the text: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MindCare AI • Nomic Embeddings + Linear SVM + Groq GPT-OSS-20B"
)

st.markdown(
    '<div class="footer-text">'
    'For research and educational use only • Not a medical diagnosis'
    '</div>',
    unsafe_allow_html=True
)