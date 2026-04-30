from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# ================= CONFIG =================
app.secret_key = "your-secret-key-123"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= DATA =================
skills_db = [
    "python", "java", "sql", "html", "css", "javascript",
    "react", "flask", "django", "aws", "docker",
    "kubernetes", "git", "github", "mysql"
]

keywords_db = [
    "teamwork", "communication", "leadership",
    "problem solving", "rest api", "debugging",
    "time management", "collaboration"
]

# ================= FUNCTIONS =================
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.lower()

# ================= DATABASE MODEL =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ================= ROUTES =================

@app.route("/")
def home():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if not user:
            flash("User not found!")
            return redirect(url_for("login"))

        if not check_password_hash(user.password, password):
            flash("Wrong password!")
            return redirect(url_for("login"))

        session["user"] = username
        return redirect(url_for("home"))

    return render_template("login.html")

# ---------- REGISTER ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        if User.query.filter_by(username=username).first():
            flash("User already exists!")
            return redirect(url_for("register"))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registered successfully! Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------- ANALYZE ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    if 'user' not in session:
        return redirect(url_for("login"))

    if 'resume' not in request.files:
        return "No file uploaded"

    file = request.files["resume"]
    jd = request.form["jd"].lower()

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    resume_text = extract_text(filepath)

    # Similarity
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_text, jd])
    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    similarity_score = round(similarity_score * 100)

    # Skills Matching
    matched = []
    missing = []

    for skill in skills_db:
        if skill in jd:
            if skill in resume_text:
                matched.append(skill)
            else:
                missing.append(skill)

    total_required = len(matched) + len(missing)
    skill_score = round((len(matched) / total_required) * 100) if total_required > 0 else 0

    match_score = round((skill_score * 0.6) + (similarity_score * 0.4))

    # Keywords
    missing_keywords = []
    for word in keywords_db:
        if word in jd and word not in resume_text:
            missing_keywords.append(word)

    # Suggestions
    suggestions = []
    for skill in missing:
        suggestions.append(f"Include {skill} in resume.")
    for word in missing_keywords:
        suggestions.append(f"Add {word} in content.")

    if not suggestions:
        suggestions.append("Your resume is well optimized.")

    # ATS Tips
    ats_tips = [
        "Use standard headings like Skills, Education, Experience.",
        "Use bullet points for projects and achievements.",
        "Mention measurable results where possible.",
        "Keep formatting clean and simple.",
        "Include keywords naturally from the JD."
    ]

    # Rating
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

# ================= RUN =================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(host="0.0.0.0", port=10000)
