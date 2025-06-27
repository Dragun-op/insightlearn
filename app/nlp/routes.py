from flask import Blueprint, render_template, request, flash, current_app
from flask_login import login_required, current_user
from app.models import db, Explanation
from app.nlp.understanding_classifier import classify_explanation
import os, requests
from app.forms import ClassifierForm

nlp = Blueprint("nlp", __name__)

HF_QG_MODEL = "google/flan-t5-base"
HF_CLF_MODEL = "facebook/bart-large-mnli"
HF_API_KEY = os.environ.get("HF_API_KEY")
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

@nlp.route("/", endpoint="index")
def index():
    return render_template("nlp/index.html")

@nlp.route("/classifier", methods=["GET", "POST"])
@login_required
def classifier():
    form = ClassifierForm()

    if form.validate_on_submit():
        topic = form.topic.data
        explanation = form.explanation.data

        if topic and not explanation:
            HF_API_KEY = current_app.config["HF_API_TOKEN"]  # ✅ works inside app context
            HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

            payload = {
                "inputs": f"Generate a clear question about: {topic}",
                "options": {"use_cache": False}
            }

            q_resp = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-base",
                headers=HEADERS,
                json=payload
            )

            if q_resp.status_code != 200:
                print("QG request failed:", q_resp.status_code, q_resp.text)
                flash("Could not generate question", "danger")
                return render_template("nlp/classifier.html", form=form)

            q_text = q_resp.json()[0]["generated_text"].strip()
            return render_template("nlp/classifier.html", form=form, topic=topic, question=q_text)

    return render_template("nlp/classifier.html", form=form)