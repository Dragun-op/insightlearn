import os
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_API_KEY = os.environ.get("HF_API_KEY")

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

CANDIDATE_LABELS = ["Understood", "Memorized", "Confused"]

def classify_explanation(text):
    if not HF_API_KEY:
        raise ValueError("Missing Hugging Face API Key")

    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": CANDIDATE_LABELS}
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")

    result = response.json()

    scores = {label: 0.0 for label in CANDIDATE_LABELS}
    for label, score in zip(result["labels"], result["scores"]):
        scores[label] = score

    prediction = result["labels"][0] 

    return {
        "text": text,
        "prediction": prediction,
        "scores": scores
    }