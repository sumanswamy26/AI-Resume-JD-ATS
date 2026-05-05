from flask import Flask, render_template, request, session, redirect, flash, send_file, jsonify
import os
import secrets
import pdfplumber
from datetime import datetime, timedelta
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ================= APP =================
app = Flask(__name__)
app.secret_key = "super-secret-key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

db = SQLAlchemy(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= DATA =================
skills_db = ["python","java","sql","html","css","javascript","react","flask","django"]
keywords_db = ["teamwork","communication","leadership","problem solving"]

# ================= HELPERS =================
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + " "
    return text.lower()

def word_tokenize(text):
    """Split text into words, handling multi-word phrases"""
    import re
    return re.findall(r'\b[\w\s]+\b', text.lower())

def contains_skill(text, skill):
    """Check if skill is present as a whole word or phrase"""
    import re
    # Create a pattern that matches the skill as a whole word or phrase
    pattern = r'\b' + re.escape(skill) + r'\b'
    return bool(re.search(pattern, text))

def generate_auto_suggestions(missing, missing_keywords):
    suggestions = []
    if missing:
        suggestions.append("Add missing skills: " + ", ".join(missing[:3]))
    if missing_keywords:
        suggestions.append("Include keywords: " + ", ".join(missing_keywords[:2]))
    suggestions.append("Use action verbs (Developed, Built)")
    suggestions.append("Add measurable achievements")
    return suggestions

def generate_ats_tips(resume_text, jd):
    tips = []
    tips.append("Use standard fonts (Arial, Calibri) for ATS compatibility")
    tips.append("Avoid graphics, images, and tables - use simple text formatting")
    tips.append("Include all relevant keywords from the job description")
    tips.append("Use standard section headings: Summary, Experience, Skills, Education")
    tips.append("List skills clearly - ATS systems scan for exact skill matches")
    tips.append("Include phone number and email in a simple format")
    tips.append("Use bullet points (•) instead of special characters")
    tips.append("Save and upload as PDF to preserve formatting")
    return tips

def generate_pdf_report(score):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph(f"Resume Score: {score}%", styles['Heading1']))
    story.append(Spacer(1, 10))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ================= MODELS =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    dark_mode = db.Column(db.Boolean, default=False)

class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    name = db.Column(db.String(200))
    best_score = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer)
    jd_title = db.Column(db.String(200))
    match_score = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ================= ROUTES =================

@app.route("/")
def home():
    if 'user' not in session:
        return redirect("/login")

    user = User.query.filter_by(username=session['user']).first()
    if not user:
        session.clear()
        return redirect("/login")

    resumes = Resume.query.filter_by(user_id=user.id).all()
    analyses = AnalysisHistory.query.all()

    return render_template("dashboard.html", resumes=resumes, analyses=analyses, dark_mode=user.dark_mode)

# LOGIN
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()

        if not user or not check_password_hash(user.password, request.form["password"]):
            flash("Invalid credentials")
            return redirect("/login")

        session["user"] = user.username
        return redirect("/")

    return render_template("login.html")

# REGISTER
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        if User.query.filter_by(username=request.form["username"]).first():
            flash("User exists")
            return redirect("/register")

        new_user = User(
            username=request.form["username"],
            password=generate_password_hash(request.form["password"])
        )
        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")

# LOGOUT
@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect("/login")

# SCORE API
@app.route("/api/score-history")
def score_history():
    analyses = AnalysisHistory.query.order_by(AnalysisHistory.created_at).all()

    return jsonify({
        "dates": [a.created_at.strftime("%m/%d") for a in analyses],
        "scores": [a.match_score for a in analyses]
    })

# ANALYZE
@app.route("/analyze", methods=["POST"])
def analyze():
    if 'user' not in session:
        return redirect("/login")

    file = request.files["resume"]
    jd = request.form["jd"].lower()
    jd_title = request.form.get("jd_title", "Untitled Job")

    user = User.query.filter_by(username=session['user']).first()

    filename = secrets.token_hex(5) + "_" + file.filename
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    resume_text = extract_text(path)

    # TF-IDF based similarity
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    try:
        vectors = tfidf.fit_transform([resume_text, jd])
        tfidf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    except Exception:
        tfidf_score = 0

    # Skill-based matching score
    matched = [s for s in skills_db if contains_skill(resume_text, s) and contains_skill(jd, s)]
    missing = [s for s in skills_db if contains_skill(jd, s) and s not in matched]

    missing_keywords = [k for k in keywords_db if contains_skill(jd, k) and not contains_skill(resume_text, k)]

    # Calculate combined score
    total_required_skills = len(matched) + len(missing)
    skill_match_score = (len(matched) / total_required_skills * 100) if total_required_skills > 0 else 0
    
    # Average the TF-IDF score and skill match score
    score = int((tfidf_score + skill_match_score) / 2)
    score = max(10, min(100, score))  # Clamp between 10-100

    if score >= 80:
        rating = "Excellent"
    elif score >= 60:
        rating = "Good"
    else:
        rating = "Needs Improvement"

    suggestions = generate_auto_suggestions(missing, missing_keywords)
    ats_tips = generate_ats_tips(resume_text, jd)

    resume = Resume(user_id=user.id, name=file.filename, best_score=score)
    db.session.add(resume)
    db.session.commit()

    analysis = AnalysisHistory(
        resume_id=resume.id,
        jd_title=jd_title,
        match_score=score
    )
    db.session.add(analysis)
    db.session.commit()

    return render_template(
        "result.html",
        score=score,
        rating=rating,
        matched=matched,
        missing=missing,
        missing_keywords=missing_keywords,
        suggestions=suggestions,
        ats_tips=ats_tips,
        interview_questions=[],
        analysis_id=analysis.id,
        dark_mode=user.dark_mode
    )

# DOWNLOAD
@app.route("/download-report/<int:id>")
def download(id):
    a = AnalysisHistory.query.get(id)

    pdf = generate_pdf_report(a.match_score)

    return send_file(pdf, as_attachment=True, download_name="report.pdf")

# ================= RUN =================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)
