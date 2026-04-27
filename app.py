from flask import Flask, render_template, request
import os
import pdfplumber
import webbrowser
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Skills database
skills_db = [
    "python", "java", "sql", "html", "css", "javascript",
    "react", "flask", "django", "aws", "docker",
    "kubernetes", "git", "github", "mysql"
]

# Keywords database
keywords_db = [
    "teamwork", "communication", "leadership",
    "problem solving", "rest api", "debugging",
    "time management", "collaboration"
]

# Extract text from PDF
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.lower()

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Analyze Resume
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["resume"]
    jd = request.form["jd"].lower()

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Extract resume text
    resume_text = extract_text(filepath)

    # Text similarity score
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_text, jd])

    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    similarity_score = round(similarity_score * 100)

    # Skill matching
    matched = []
    missing = []

    for skill in skills_db:
        if skill in jd:
            if skill in resume_text:
                matched.append(skill)
            else:
                missing.append(skill)

    # Skill score
    total_required = len(matched) + len(missing)

    if total_required > 0:
        skill_score = round((len(matched) / total_required) * 100)
    else:
        skill_score = 0

    # Final ATS score
    match_score = round((skill_score * 0.6) + (similarity_score * 0.4))

    # Missing keywords
    missing_keywords = []

    for word in keywords_db:
        if word in jd and word not in resume_text:
            missing_keywords.append(word)

    # Suggestions
    suggestions = []

    for skill in missing:
        suggestions.append(
            f"Include {skill} in Skills, Projects, or Experience section."
        )

    for word in missing_keywords:
        suggestions.append(
            f"Add evidence of {word} in resume content."
        )

    if not suggestions:
        suggestions.append(
            "Your resume aligns well with the job description."
        )

    # ATS Tips
    ats_tips = [
        "Use standard headings like Skills, Education, Experience.",
        "Use bullet points for projects and achievements.",
        "Mention measurable results where possible.",
        "Keep formatting clean and simple.",
        "Include keywords naturally from the JD."
    ]

    # Rating label
    if match_score >= 85:
        rating = "Excellent Match"
    elif match_score >= 70:
        rating = "Good Match"
    elif match_score >= 50:
        rating = "Average Match"
    else:
        rating = "Needs Improvement"

    return render_template(
        "result.html",
        score=match_score,
        rating=rating,
        matched=matched,
        missing=missing,
        missing_keywords=missing_keywords,
        suggestions=suggestions,
        ats_tips=ats_tips
    )

# Auto open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

# Run App
if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True)