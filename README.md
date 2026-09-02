# 🧠 MindCare AI — Mental Health Support Assistant

MindCare AI is a mental-health NLP project that classifies Reddit-style text into one of five mental-health-related categories and generates a concise, supportive response using a Large Language Model (LLM).

The system combines:

- Nomic Embed Text for semantic text embeddings
- Linear SVM for classification
- A keyword-based high-risk safety layer
- Groq API with OpenAI GPT-OSS-20B for response generation
- Streamlit for the user interface

> ⚠️ This project is intended for educational and research purposes. It does not provide medical diagnoses or replace professional mental-health care.

---

## 🎯 Project Objective

The objective is to build a system that can:

1. Accept a Reddit-style mental-health post as input.
2. Convert the text into semantic embeddings.
3. Classify the text into one of five categories.
4. Detect predefined high-risk/self-harm related language.
5. Generate a structured and supportive response using an LLM.

---

## 🏷️ Classification Categories

| Class | Category |
|------:|----------|
| 0 | Stress |
| 1 | Depression |
| 2 | Bipolar Disorder |
| 3 | Personality Disorder |
| 4 | Anxiety |

---
## Project Folder Structure
```text
mental-health-support-assistant/
│
├── app.py
├── mental_health_svm_artifacts.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```
secrets.toml is a local secret file that should include the Groq API Key in this format : GROQ_API_KEY = " "

## 🏗️ System Architecture

```text
                    User Input
                        │
                        ▼
                 Text Preprocessing
                        │
                        ▼
                Nomic Embeddings
                        │
                        ▼
                  Linear SVM
                        │
                        ▼
                Predicted Category
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
     High-Risk Detector      Original Text
             │                     │
             └──────────┬──────────┘
                        ▼
                 Groq GPT-OSS-20B
                        │
                        ▼
               Supportive Response






