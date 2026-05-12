from flask import Flask, render_template, request, session, redirect, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import text

import os
import json
import re
import secrets
import pdfplumber
import matplotlib.pyplot as plt
import urllib.error
import urllib.request

from io import BytesIO
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from functools import wraps

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper

# ================= APP =================
app = Flask(__name__)

app.secret_key = "super-secret-key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

db = SQLAlchemy(app)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ================= UNIVERSAL SKILLS =================
GLOBAL_SKILLS = [

    # FRONTEND
    "html", "css", "javascript", "typescript",
    "react", "next.js", "nextjs", "vue", "angular",
    "tailwind", "bootstrap", "redux",

    # BACKEND
    "node.js", "nodejs", "express", "express.js",
    "django", "flask", "fastapi", "spring boot",
    "laravel", ".net", "graphql", "rest api",

    # DATABASE
    "mysql", "postgresql", "mongodb", "firebase",
    "redis", "sqlite", "oracle", "supabase",

    # CLOUD / DEVOPS
    "aws", "azure", "gcp", "docker", "kubernetes",
    "jenkins", "terraform", "ci/cd", "nginx",

    # AI / ML
    "machine learning", "deep learning",
    "tensorflow", "pytorch", "langchain",
    "llm", "nlp", "computer vision",

    # PROGRAMMING
    "python", "java", "c", "c++", "c#",
    "go", "rust", "php", "ruby",
    "kotlin", "swift",

    # TOOLS
    "git", "github", "figma",
    "postman", "jira", "linux",
    "vercel", "netlify",

    # SOFT SKILLS
    "communication", "leadership",
    "teamwork", "problem solving",
    "analytical", "creative"
]


# ================= HELPERS =================
def extract_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            t = page.extract_text()

            if t:
                text += t + " "

    return text.lower()


def normalize_text(text):

    cleaned = re.sub(
        r"[^a-z0-9+#.\s-]",
        " ",
        text.lower()
    )

    return re.sub(r"\s+", " ", cleaned).strip()


def extract_skills(text):

    found_skills = []

    text = text.lower()

    for skill in GLOBAL_SKILLS:

        pattern = r'\b' + re.escape(skill.lower()) + r'\b'

        if re.search(pattern, text):
            found_skills.append(skill)

    return list(set(found_skills))


# ================= AUTO SUGGESTIONS =================
def generate_auto_suggestions(missing, missing_keywords):

    suggestions = []

    if missing:

        suggestions.append(
            "Add missing skills like: " +
            ", ".join(missing[:5])
        )

    if missing_keywords:

        suggestions.append(
            "Add important keywords like: " +
            ", ".join(missing_keywords[:5])
        )

    suggestions.append(
        "Use strong action verbs like Developed, Built, Created"
    )

    suggestions.append(
        "Add measurable achievements with numbers"
    )

    suggestions.append(
        "Keep formatting simple and ATS-friendly"
    )

    return suggestions


# ================= ATS TIPS =================
def generate_ats_tips():

    return [

        "Use ATS-friendly fonts like Arial or Calibri",

        "Avoid tables, graphics, and images",

        "Use standard section headings",

        "Include relevant job keywords",

        "Keep resume formatting simple",

        "Use bullet points for achievements",

        "Save resume as PDF"
    ]


def extract_job_info(jd_text):
    """Extract job title and company from job description"""
    
    lines = jd_text.split('\n')[:15]  # Check first 15 lines
    job_title = "Untitled Position"
    company_name = "Unknown Company"
    
    for i, line in enumerate(lines):
        original_line = line
        line = line.strip().lower()
        
        # Extract company
        if 'company:' in line or line.startswith('company '):
            # Remove the "company:" label and get the name
            for sep in [':', '-', '|']:
                if sep in original_line:
                    company_name = original_line.split(sep, 1)[1].strip()
                    break
            if company_name.lower().startswith('company'):
                company_name = "Unknown Company"
        
        # Extract job title  
        if 'position:' in line or 'job title:' in line or 'role:' in line:
            for sep in [':', '-', '|']:
                if sep in original_line:
                    job_title = original_line.split(sep, 1)[1].strip()
                    break
        elif i == 1 and len(original_line) > 5 and len(original_line) < 100:
            # If second line is short, it might be the job title
            if not any(x in line for x in ['at', 'for', 'the', 'about']):
                job_title = original_line.strip()
    
    # Clean up extracted values
    company_name = company_name[:100].strip() if company_name else "Unknown Company"
    job_title = job_title[:100].strip() if job_title else "Untitled Position"
    
    return job_title, company_name


def get_fit_level(score):
    """Determine fit level based on ATS score"""
    
    if score >= 85:
        return "Perfect Fit"
    elif score >= 75:
        return "Strong Fit"
    elif score >= 60:
        return "Good Fit"
    elif score >= 45:
        return "Moderate Fit"
    else:
        return "Needs Work"


def get_ai_provider(user=None):

    if user and user.ai_provider in {"openai", "gemini", "local"}:
        return user.ai_provider

    provider = os.getenv("HIRELENS_AI_PROVIDER", "auto").strip().lower()

    if provider in {"openai", "gemini"}:
        return provider

    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    if os.getenv("GEMINI_API_KEY"):
        return "gemini"

    return "local"


def get_provider_api_key(user, provider):

    if provider == "openai":
        if user and user.openai_api_key:
            return user.openai_api_key
        return os.getenv("OPENAI_API_KEY")

    if provider == "gemini":
        if user and user.gemini_api_key:
            return user.gemini_api_key
        return os.getenv("GEMINI_API_KEY")

    return None


def call_openai(messages, api_key):

    if not api_key:
        return None

    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": messages,
        "temperature": 0.4,
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"].strip()


def call_gemini(messages, api_key):

    if not api_key:
        return None

    system_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    conversation = []

    for message in messages:
        if message["role"] == "system":
            continue

        role = "model" if message["role"] == "assistant" else "user"
        conversation.append({"role": role, "parts": [{"text": message["content"]}]})

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": conversation,
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 350},
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')}:generateContent?key={api_key}"
    )

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    candidates = data.get("candidates", [])

    if not candidates:
        return None

    parts = candidates[0].get("content", {}).get("parts", [])

    return "".join(part.get("text", "") for part in parts).strip() or None


def generate_resume_recommendations(score, matched, missing, missing_keywords):

    recommendations = []

    if score >= 80:
        recommendations.append("Your resume is already competitive. Tighten wording and highlight measurable outcomes.")
    elif score >= 60:
        recommendations.append("You are close. Add missing skills and align more of your bullets with the job description.")
    else:
        recommendations.append("The biggest lift will come from matching the core skills in the job description.")

    if missing:
        recommendations.append(
            "Add explicit mentions of: " + ", ".join(missing[:5])
        )

    if missing_keywords:
        recommendations.append(
            "Weave in keywords naturally such as: " + ", ".join(missing_keywords[:5])
        )

    if matched:
        recommendations.append(
            "Make the matched skills more visible in your summary and top bullet points: " + ", ".join(matched[:4])
        )

    recommendations.extend([
        "Start bullets with action verbs and include numbers where possible.",
        "Keep formatting simple so ATS systems can parse it reliably."
    ])

    return recommendations


def build_chat_system_prompt(context):

    score = context.get("score")
    rating = context.get("rating")
    matched = context.get("matched", [])
    missing = context.get("missing", [])
    missing_keywords = context.get("missing_keywords", [])

    analysis_lines = [
        "You are HireLens Assistant, a concise resume and ATS optimization coach.",
        "Answer with practical advice and short paragraphs.",
        "If the user asks about analysis, use the provided context.",
    ]

    if score is not None:
        analysis_lines.append(f"Current ATS score: {score}.")
        analysis_lines.append(f"Rating: {rating}.")

    if matched:
        analysis_lines.append("Matched skills: " + ", ".join(matched[:10]))

    if missing:
        analysis_lines.append("Missing skills: " + ", ".join(missing[:10]))

    if missing_keywords:
        analysis_lines.append("Missing keywords: " + ", ".join(missing_keywords[:10]))

    analysis_lines.append(
        "When suggesting improvements, prioritize the missing skills and keywords, then recommend measurable bullets and ATS-friendly formatting."
    )

    return "\n".join(analysis_lines)


def get_chat_history(user_id, limit=12):

    history = ChatMessage.query.filter_by(
        user_id=user_id
    ).order_by(
        ChatMessage.created_at.asc()
    ).all()

    return [
        {
            "id": item.id,
            "role": item.role,
            "analysis_id": item.analysis_id,
            "message": item.message,
            "created_at": item.created_at.isoformat()
        }
        for item in history[-limit:]
    ]


def save_chat_message(user_id, role, message, analysis_id=None):

    chat_message = ChatMessage(
        user_id=user_id,
        analysis_id=analysis_id,
        role=role,
        message=message[:4000]
    )

    db.session.add(chat_message)
    db.session.commit()

    return chat_message


def generate_chatbot_reply(message, analysis_context=None):

    analysis_context = analysis_context or {}

    text = (message or "").strip()
    lower_text = text.lower()

    score = analysis_context.get("score")
    rating = analysis_context.get("rating")
    matched = analysis_context.get("matched", [])
    missing = analysis_context.get("missing", [])
    missing_keywords = analysis_context.get("missing_keywords", [])
    suggestions = analysis_context.get("suggestions", [])
    recommendations = analysis_context.get("recommendations", [])

    if not lower_text:
        return (
            "Ask me anything about resumes, ATS, bullet rewriting, interview prep, cover letters, "
            "or career planning."
        )

    if any(word in lower_text for word in ["hello", "hi", "hey", "start", "help"]):
        return (
            "You can ask anything resume-related here: rewrite bullets, improve summary, prepare interview answers, "
            "explain ATS score, or plan upskilling."
        )

    if any(word in lower_text for word in ["upload", "paste", "how do i use", "how it works", "analyze"]):
        return (
            "Upload a PDF resume, paste the job description, and click Analyze Resume. "
            "Then ask me to improve bullets, summary, or keyword usage based on your result."
        )

    if any(word in lower_text for word in ["score", "rating", "result", "results", "how did i do"]):
        if score is None:
            return "Run an analysis first and I can explain your ATS score and what to fix first."

        matched_count = len(matched) if matched else 0
        missing_count = len(missing) if missing else 0
        keywords_count = len(missing_keywords) if missing_keywords else 0

        result = f"📊 **Your ATS Score: {score}%** ({rating} rating)\n\n"
        result += f"✓ Matched Skills: {matched_count}\n"
        result += f"⚠️  Missing Skills: {missing_count}\n"
        result += f"⚠️  Missing Keywords: {keywords_count}\n\n"
        result += "Ask me about missing skills or improvements to get specific recommendations!"

        return result

    # Enhanced: Missing Skills & Suggestions
    if any(word in lower_text for word in ["missing", "what skills", "what's missing", "gap", "skills gap"]):
        if not missing and not missing_keywords:
            return "Great news! Your resume already covers the target skills well. Focus on strengthening your bullet descriptions now."

        result = "🎯 **Current Analysis**\n\n"

        if missing:
            result += "**Missing Skills:**\n"
            for skill in missing[:8]:
                result += f"  • {skill}\n"
            result += "\n"

        if missing_keywords:
            result += "**Missing Keywords:**\n"
            for kw in missing_keywords[:8]:
                result += f"  • {kw}\n"
            result += "\n"

        result += "💡 **How to Add These:**\n"
        result += "1. Review your experience for past projects using these tools\n"
        result += "2. Rewrite bullets to explicitly mention the technology or skill\n"
        result += "3. Add these keywords naturally (avoid keyword stuffing)\n\n"
        result += "Ask me to 'rewrite my bullet' or 'improve my summary' for specific help!"

        return result

    # Enhanced: Improvement suggestions
    if any(word in lower_text for word in ["improve", "better", "optimize", "fix", "recommendation", "how do i improve"]):
        if score is None:
            return "Run an analysis first and I can give you specific improvement recommendations."

        result = "✨ **Resume Improvement Roadmap**\n\n"

        if score and score < 80:
            result += "**Step 1: Address Skills Gap**\n"
            if missing:
                result += f"  Add these top skills: {', '.join(missing[:4])}\n"
            if missing_keywords:
                result += f"  Weave in keywords: {', '.join(missing_keywords[:4])}\n"
            result += "\n"

        result += "**Step 2: Strengthen Your Bullets**\n"
        result += "  • Use format: [Action] + [Tool/Tech] + [Measurable Result]\n"
        result += "  • Example: 'Reduced API response time by 40% using Redis caching'\n"
        result += "\n"

        if matched:
            result += "**Step 3: Highlight Matched Skills**\n"
            result += f"  • Feature these in your summary and top bullets: {', '.join(matched[:4])}\n"
            result += "\n"

        if recommendations:
            result += "**Recommended Actions:**\n"
            for i, rec in enumerate(recommendations[:4], 1):
                result += f"  {i}. {rec}\n"

        return result

    if any(word in lower_text for word in ["summary", "professional summary", "about me"]):
        return (
            "Use a 3-line summary: role target, years/domain strengths, and top measurable impact. "
            "If you share your current summary, I can rewrite it for ATS and recruiters."
        )

    if any(word in lower_text for word in ["rewrite", "bullet", "experience", "project"]):
        return (
            "Share one bullet and I will rewrite it into a stronger ATS-friendly version. "
            "Template: Built [what] using [tools], resulting in [metric/impact]."
        )

    if any(word in lower_text for word in ["cover letter", "email", "application message"]):
        return (
            "I can draft a short, role-specific cover letter. Share job title, company, and 2-3 achievements to include."
        )

    if any(word in lower_text for word in ["interview", "question", "hr round", "technical round"]):
        return (
            "I can generate likely interview questions and strong sample answers for your role. "
            "Tell me the job title and level (fresher, mid, senior)."
        )

    if any(word in lower_text for word in ["career", "switch", "roadmap", "learn", "upskill"]):
        return (
            "I can create a 30-60-90 day upskilling roadmap aligned to your target role and missing skills."
        )

    if any(word in lower_text for word in ["download", "report", "pdf"]):
        return "Use the Download Full Report button to export your analysis as PDF."

    if any(word in lower_text for word in ["skill", "skills", "keyword", "keywords"]):
        if missing:
            result = "**Top Skills to Add:**\n"
            for skill in missing[:6]:
                result += f"  • {skill}\n"
            result += "\nAsk 'what skills am I missing' or 'how do I improve' for detailed guidance!"
            return result
        return (
            "I track matched skills, missing skills, and missing keywords. If you want, I can prioritize the top 5 highest-impact additions first."
        )

    return (
        "I can answer broader resume questions too. Ask for bullet rewrites, summary rewrite, cover letter draft, "
        "interview Q&A prep, role switch plan, or ATS optimization strategy."
    )


@app.route("/chatbot", methods=["POST"])
def chatbot():

    if "user" not in session:
        return jsonify({"reply": "Please log in to use the assistant."}), 401

    data = request.get_json(silent=True) or {}

    message = data.get("message", "")

    context = data.get("context") or session.get("last_analysis", {})

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        return jsonify({"reply": "Please log in again to use the assistant."}), 401

    save_chat_message(user.id, "user", message, context.get("analysis_id"))

    history = get_chat_history(user.id)

    messages = [
        {"role": "system", "content": build_chat_system_prompt(context)}
    ]

    for item in history:
        messages.append({
            "role": "assistant" if item["role"] == "assistant" else "user",
            "content": item["message"]
        })

    provider = get_ai_provider(user)
    provider_api_key = get_provider_api_key(user, provider)

    reply = None

    try:
        if provider == "openai":
            reply = call_openai(messages, provider_api_key)
        elif provider == "gemini":
            reply = call_gemini(messages, provider_api_key)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError):
        reply = None

    if not reply:
        reply = generate_chatbot_reply(message, context)

        if provider in {"openai", "gemini"} and not provider_api_key:
            reply = (
                f"{reply}\n\nNote: {provider.title()} is selected but no API key is saved. "
                "Add your key in Settings for full live AI answers."
            )

    save_chat_message(user.id, "assistant", reply, context.get("analysis_id"))

    return jsonify({
        "reply": reply,
        "quick_replies": [
            "Rewrite my summary",
            "Rewrite one bullet point",
            "Create interview questions",
            "How can I improve my ATS score?"
        ],
        "history": get_chat_history(user.id)
    })


@app.route("/chatbot/history", methods=["GET"])
def chatbot_history():

    if "user" not in session:
        return jsonify({"history": []}), 401

    user = User.query.filter_by(username=session['user']).first()

    if not user:
        return jsonify({"history": []}), 401

    return jsonify({"history": get_chat_history(user.id, limit=200)})


@app.route("/chat-history")
@login_required
def chat_history_page():

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    messages = ChatMessage.query.filter_by(
        user_id=user.id
    ).order_by(
        ChatMessage.created_at.desc()
    ).all()

    return render_template(
        "chat_history.html",
        user=user,
        chat_messages=messages,
        dark_mode=user.dark_mode,
        chatbot_context=session.get("last_analysis", {}),
        chatbot_history=get_chat_history(user.id)
    )


@app.route("/chat/export")
@login_required
def chat_export():

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    messages = get_chat_history(user.id, limit=500)

    payload = {
        "user": user.username,
        "exported_at": datetime.utcnow().isoformat(),
        "messages": messages
    }

    buffer = BytesIO(json.dumps(payload, indent=2).encode("utf-8"))

    return send_file(
        buffer,
        as_attachment=True,
        download_name="hirelens_chat_history.json",
        mimetype="application/json"
    )


@app.route("/chat/delete/<int:id>", methods=["POST"])
@login_required
def chat_delete(id):

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    message = ChatMessage.query.filter_by(
        id=id,
        user_id=user.id
    ).first()

    if message:
        db.session.delete(message)
        db.session.commit()

    return redirect("/chat-history")


@app.route("/chat/delete-all", methods=["POST"])
@login_required
def chat_delete_all():

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    ChatMessage.query.filter_by(user_id=user.id).delete()
    db.session.commit()

    return redirect("/chat-history")


# ================= GRAPH =================
def create_score_graph(score):

    labels = ["Matched", "Missing"]

    values = [score, 100 - score]

    plt.figure(figsize=(4, 3))

    plt.bar(labels, values)

    plt.title("ATS Score Breakdown")

    graph_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "score_graph.png"
    )

    plt.savefig(graph_path, bbox_inches='tight')

    plt.close()

    return graph_path


# ================= PDF REPORT =================
def generate_pdf_report(
    score,
    rating,
    suggestions,
    ats_tips
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    story = []

    # TITLE
    story.append(
        Paragraph(
            "ATS Resume Analysis Report",
            styles['Title']
        )
    )

    story.append(Spacer(1, 20))

    # SCORE
    story.append(
        Paragraph(
            f"<b>ATS Score:</b> {score}%",
            styles['Heading2']
        )
    )

    story.append(
        Paragraph(
            f"<b>Rating:</b> {rating}",
            styles['BodyText']
        )
    )

    story.append(Spacer(1, 20))

    # GRAPH
    graph = create_score_graph(score)

    story.append(
        Paragraph(
            "Score Breakdown Graph",
            styles['Heading3']
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Image(graph, width=300, height=180)
    )

    story.append(Spacer(1, 20))

    # SUGGESTIONS
    story.append(
        Paragraph(
            "Suggestions",
            styles['Heading3']
        )
    )

    for s in suggestions:

        story.append(
            Paragraph(
                f"• {s}",
                styles['BodyText']
            )
        )

    story.append(Spacer(1, 20))

    # ATS TIPS
    story.append(
        Paragraph(
            "ATS Optimization Tips",
            styles['Heading3']
        )
    )

    for t in ats_tips:

        story.append(
            Paragraph(
                f"• {t}",
                styles['BodyText']
            )
        )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by AI ATS Resume Analyzer",
            styles['Italic']
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# ================= MODELS =================
class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True
    )

    password = db.Column(
        db.String(200)
    )

    dark_mode = db.Column(
        db.Boolean,
        default=False
    )

    ai_provider = db.Column(
        db.String(20),
        default="local"
    )

    openai_api_key = db.Column(
        db.String(255)
    )

    gemini_api_key = db.Column(
        db.String(255)
    )


class Resume(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(db.Integer)

    name = db.Column(
        db.String(200)
    )

    best_score = db.Column(
        db.Integer,
        default=0
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class AnalysisHistory(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    resume_id = db.Column(db.Integer)

    jd_title = db.Column(
        db.String(200)
    )

    company_name = db.Column(
        db.String(200)
    )

    fit_level = db.Column(
        db.String(50)
    )

    match_score = db.Column(
        db.Integer
    )

    matched = db.Column(db.Text)

    missing = db.Column(db.Text)

    keywords = db.Column(db.Text)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class ChatMessage(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(db.Integer)

    analysis_id = db.Column(db.Integer)

    role = db.Column(
        db.String(20)
    )

    message = db.Column(db.Text)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# ================= HOME =================
@app.route("/")
def home():

    if 'user' not in session:
        return redirect("/login")

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    return render_template(
        "index.html",
        dark_mode=user.dark_mode,
        chatbot_context=session.get("last_analysis", {}),
        chatbot_history=get_chat_history(user.id)
    )


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():

    if 'user' not in session:
        return redirect("/login")

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    # USER RESUMES
    resumes = Resume.query.filter_by(
        user_id=user.id
    ).order_by(
        Resume.created_at.desc()
    ).all()

    # Attach latest analysis info to each resume for dashboard display
    for resume in resumes:
        latest = AnalysisHistory.query.filter_by(
            resume_id=resume.id
        ).order_by(
            AnalysisHistory.created_at.desc()
        ).first()

        if latest:
            # matched/missing stored as comma-joined strings
            matched_list = latest.matched.split(",") if latest.matched else []
            missing_list = latest.missing.split(",") if latest.missing else []

            # expose friendly attributes for templates
            resume.latest_analysis = latest
            resume.latest_matched = [m for m in matched_list if m]
            resume.latest_missing = [m for m in missing_list if m]
            resume.analysis_status = (
                "Good" if len(resume.latest_missing) == 0 else "Missing Skills"
            )

        else:
            resume.latest_analysis = None
            resume.latest_matched = []
            resume.latest_missing = []
            resume.analysis_status = "Not Analyzed"

    # USER ANALYSES
    analyses = AnalysisHistory.query.join(
        Resume,
        Resume.id == AnalysisHistory.resume_id
    ).filter(
        Resume.user_id == user.id
    ).order_by(
        AnalysisHistory.created_at.desc()
    ).all()

    # ADD RATING TO ANALYSIS
    for analysis in analyses:

        if analysis.match_score >= 80:
            analysis.rating = "Excellent"

        elif analysis.match_score >= 60:
            analysis.rating = "Good"

        else:
            analysis.rating = "Needs Improvement"

    return render_template(
        "dashboard.html",
        user=user,
        resumes=resumes,
        analyses=analyses,
        dark_mode=user.dark_mode,
        chatbot_context=session.get("last_analysis", {}),
        chatbot_history=get_chat_history(user.id)
    )


# ================= ANALYSIS DETAIL =================
@app.route("/analysis/<int:id>")
@login_required
def analysis(id):

    user = User.query.filter_by(username=session['user']).first()

    if not user:
        session.clear()
        return redirect("/login")

    a = AnalysisHistory.query.get(id)

    if not a:
        flash("Analysis not found")
        return redirect("/dashboard")

    resume = Resume.query.get(a.resume_id)

    if not resume or resume.user_id != user.id:
        flash("Unauthorized access to analysis")
        return redirect("/dashboard")

    matched = a.matched.split(",") if a.matched else []
    missing = a.missing.split(",") if a.missing else []
    missing_keywords = a.keywords.split(",") if a.keywords else []

    rating = (
        "Excellent" if a.match_score >= 80 else
        "Good" if a.match_score >= 60 else
        "Needs Improvement"
    )

    company_name = a.company_name if hasattr(a, 'company_name') and a.company_name else "Unknown Company"
    fit_level = a.fit_level if hasattr(a, 'fit_level') and a.fit_level else get_fit_level(a.match_score)

    suggestions = generate_auto_suggestions(missing, missing_keywords)
    ats_tips = generate_ats_tips()
    recommendations = generate_resume_recommendations(
        a.match_score, matched, missing, missing_keywords
    )

    return render_template(
        "analysis.html",
        user=user,
        resume=resume,
        analysis=a,
        matched=[m for m in matched if m],
        missing=[m for m in missing if m],
        missing_keywords=[k for k in missing_keywords if k],
        rating=rating,
        company_name=company_name,
        fit_level=fit_level,
        suggestions=suggestions,
        ats_tips=ats_tips,
        recommendations=recommendations,
        dark_mode=user.dark_mode,
        chatbot_context=session.get("last_analysis", {}),
        chatbot_history=get_chat_history(user.id)
    )


# ================= PROFILE =================
@app.route("/profile")
def profile():

    if 'user' not in session:
        return redirect("/login")

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    # USER RESUMES
    resumes = Resume.query.filter_by(
        user_id=user.id
    ).all()

    # USER ANALYSES
    analyses = AnalysisHistory.query.join(
        Resume,
        Resume.id == AnalysisHistory.resume_id
    ).filter(
        Resume.user_id == user.id
    ).all()

    total_resumes = len(resumes)

    total_analyses = len(analyses)

    # BEST SCORE
    if resumes:
        best_score = max(
            r.best_score for r in resumes
        )
    else:
        best_score = 0

    # AVERAGE SCORE
    if analyses:

        average_score = int(
            sum(a.match_score for a in analyses)
            / len(analyses)
        )

    else:
        average_score = 0

    return render_template(
        "profile.html",
        user=user,
        total_resumes=total_resumes,
        total_analyses=total_analyses,
        best_score=best_score,
        average_score=average_score,
        dark_mode=user.dark_mode,
        chatbot_context=session.get("last_analysis", {}),
        chatbot_history=get_chat_history(user.id)
    )


# ================= DELETE RESUME =================
@app.route("/delete-resume/<int:id>", methods=["POST"])
def delete_resume(id):

    if 'user' not in session:
        return redirect("/login")

    resume = Resume.query.get(id)

    if resume:

        # DELETE ANALYSIS HISTORY
        AnalysisHistory.query.filter_by(
            resume_id=resume.id
        ).delete()

        db.session.delete(resume)

        db.session.commit()

    return redirect("/dashboard")


# ================= DELETE ANALYSIS =================
@app.route("/delete-analysis/<int:id>", methods=["POST"])
def delete_analysis(id):

    if 'user' not in session:
        return redirect("/login")

    analysis = AnalysisHistory.query.get(id)

    if analysis:

        db.session.delete(analysis)

        db.session.commit()

    return redirect("/dashboard")
# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        user = User.query.filter_by(
            username=request.form["username"]
        ).first()

        if not user or not check_password_hash(
            user.password,
            request.form["password"]
        ):

            flash("Invalid credentials")

            return redirect("/login")

        session["user"] = user.username

        return redirect("/")

    return render_template("login.html")


# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        existing = User.query.filter_by(
            username=request.form["username"]
        ).first()

        if existing:

            flash("User already exists")

            return redirect("/register")

        user = User(
            username=request.form["username"],
            password=generate_password_hash(
                request.form["password"]
            )
        )

        db.session.add(user)

        db.session.commit()

        return redirect("/login")

    return render_template("register.html")





# ================= LOGOUT =================
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():

    user = User.query.filter_by(username=session['user']).first()

    if request.method == "POST":

        # checkbox returns "on" if checked, None if not
        user.dark_mode = "dark_mode" in request.form

        ai_provider = request.form.get("ai_provider", "local").strip().lower()
        if ai_provider not in {"local", "openai", "gemini"}:
            ai_provider = "local"
        user.ai_provider = ai_provider

        openai_api_key = request.form.get("openai_api_key", "").strip()
        gemini_api_key = request.form.get("gemini_api_key", "").strip()

        if openai_api_key:
            user.openai_api_key = openai_api_key

        if gemini_api_key:
            user.gemini_api_key = gemini_api_key

        if request.form.get("clear_openai_key") == "1":
            user.openai_api_key = None

        if request.form.get("clear_gemini_key") == "1":
            user.gemini_api_key = None

        db.session.commit()

        flash("Settings updated successfully")
        return redirect("/settings")

    return render_template(
        "settings.html",
        user=user,
        dark_mode=user.dark_mode,
        chatbot_context=session.get("last_analysis", {}),
        chatbot_history=get_chat_history(user.id)
    )


# ================= ANALYZE =================
@app.route("/analyze", methods=["POST"])
def analyze():

    if 'user' not in session:
        return redirect("/login")

    try:
        if "resume" not in request.files or request.files["resume"].filename == '':
            flash("Please upload a resume PDF", "error")
            return redirect("/")
        
        file = request.files["resume"]
        
        if not file.filename.lower().endswith('.pdf'):
            flash("Please upload a PDF file", "error")
            return redirect("/")
        
        if "jd" not in request.form or not request.form["jd"].strip():
            flash("Please paste a job description", "error")
            return redirect("/")
        
        # Extract company info from raw job description before normalization
        raw_jd = request.form["jd"]
        extracted_title, company_name = extract_job_info(raw_jd)
        
        jd = normalize_text(
            raw_jd
        )

        jd_title = request.form.get(
            "jd_title",
            "Untitled Job"
        )

        user = User.query.filter_by(
            username=session['user']
        ).first()
        
        if not user:
            session.clear()
            return redirect("/login")

        # SAVE FILE
        filename = secrets.token_hex(5) + "_" + file.filename

        path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(path)

        # EXTRACT TEXT
        resume_text = normalize_text(
            extract_text(path)
        )

        # TFIDF SCORE
        tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            stop_words='english'
        )

        try:

            vectors = tfidf.fit_transform([
                resume_text,
                jd
            ])

            tfidf_score = cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )[0][0] * 100

        except Exception:

            tfidf_score = 0

        # EXTRACT SKILLS
        resume_skills = extract_skills(
            resume_text
        )

        jd_skills = extract_skills(
            jd
        )

        matched = []

        missing = []

        for skill in jd_skills:

            if skill in resume_skills:
                matched.append(skill)

            else:
                missing.append(skill)

        # KEYWORDS
        resume_words = set(
            resume_text.split()
        )

        jd_words = set(
            jd.split()
        )

        missing_keywords = []

        for word in jd_words:

            if (
                len(word) > 4
                and word not in resume_words
                and word not in GLOBAL_SKILLS
            ):

                missing_keywords.append(word)

        missing_keywords = list(
            set(missing_keywords)
        )[:10]

        # ================= BETTER ATS SCORE =================

        # SKILL SCORE
        if len(jd_skills) > 0:

            skill_score = (
                len(matched) / len(jd_skills)
            ) * 100

        else:

            skill_score = 0

        # KEYWORD SCORE
        matched_keywords = len(jd_words) - len(missing_keywords)

        keyword_score = (
            matched_keywords / max(len(jd_words), 1)
        ) * 100

        keyword_score = max(
            0,
            min(100, keyword_score)
        )

        # FINAL SCORE
        score = int(round(

            (skill_score * 0.70) +

            (keyword_score * 0.15) +

            (tfidf_score * 0.15)

        ))

        # STRICT LIMITS
        if len(matched) <= 2:
            score = min(score, 35)

        if len(matched) == 0:
            score = min(score, 15)

        score = max(0, min(100, score))

        # ================= RATING =================
        if score >= 80:

            rating = "Excellent"

        elif score >= 60:

            rating = "Good"

        else:

            rating = "Needs Improvement"

        # ================= SUGGESTIONS =================
        suggestions = generate_auto_suggestions(
            missing,
            missing_keywords
        )

        ats_tips = generate_ats_tips()

        recommendations = generate_resume_recommendations(
            score,
            matched,
            missing,
            missing_keywords
        )

        # GET FIT LEVEL
        fit_level = get_fit_level(score)

        # SAVE RESUME
        resume = Resume(
            user_id=user.id,
            name=file.filename,
            best_score=score
        )

        db.session.add(resume)

        db.session.commit()

        # SAVE ANALYSIS
        analysis = AnalysisHistory(
            resume_id=resume.id,
            jd_title=jd_title,
            company_name=company_name,
            fit_level=fit_level,
            match_score=score,
            matched=",".join(matched),
            missing=",".join(missing),
            keywords=",".join(missing_keywords)
        )

        db.session.add(analysis)

        db.session.commit()

        session["last_analysis"] = {
            "score": score,
            "rating": rating,
            "matched": matched,
            "missing": missing,
            "missing_keywords": missing_keywords,
            "recommendations": recommendations,
            "analysis_id": analysis.id,
            "company_name": company_name,
            "fit_level": fit_level
        }

        return render_template(
            "result.html",

            score=score,

            rating=rating,

            matched=matched,

            missing=missing,

            missing_keywords=missing_keywords,

            suggestions=suggestions,

            recommendations=recommendations,

            ats_tips=ats_tips,

            analysis_id=str(analysis.id),

            company_name=company_name,

            fit_level=fit_level,

            dark_mode=user.dark_mode,
            chatbot_context=session.get("last_analysis", {}),
            chatbot_history=get_chat_history(user.id)
        )

    except Exception as e:
        flash(f"Error analyzing resume: {str(e)}", "error")
        return redirect("/")


# ================= DOWNLOAD REPORT =================
@app.route("/download-report/<int:id>")
def download(id):

    a = AnalysisHistory.query.get(id)

    if not a:
        return "Analysis not found"

    rating = (
        "Excellent"
        if a.match_score >= 80
        else "Good"
        if a.match_score >= 60
        else "Needs Improvement"
    )

    suggestions = [
        "Add more relevant technical skills",
        "Improve keyword optimization",
        "Use stronger action verbs",
        "Add measurable achievements"
    ]

    ats_tips = generate_ats_tips()

    pdf = generate_pdf_report(
        a.match_score,
        rating,
        suggestions,
        ats_tips
    )

    return send_file(
        pdf,
        as_attachment=True,
        download_name="ATS_Report.pdf"
    )


def ensure_schema_updates():

    if not app.config['SQLALCHEMY_DATABASE_URI'].startswith("sqlite"):
        return

    with db.engine.begin() as connection:
        user_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(user)"))
        }

        if "ai_provider" not in user_columns:
            connection.execute(text("ALTER TABLE user ADD COLUMN ai_provider VARCHAR(20) DEFAULT 'local'"))

        if "openai_api_key" not in user_columns:
            connection.execute(text("ALTER TABLE user ADD COLUMN openai_api_key VARCHAR(255)"))

        if "gemini_api_key" not in user_columns:
            connection.execute(text("ALTER TABLE user ADD COLUMN gemini_api_key VARCHAR(255)"))


# ================= RUN =================
if __name__ == "__main__":

    with app.app_context():
        db.create_all()
        ensure_schema_updates()

    app.run(debug=True)