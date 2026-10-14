import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
COLUMNS_PATH = BASE_DIR / "models" / "columns.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

st.set_page_config(page_title="EduPredict | Indian College Student Performance Predictor", page_icon="🎓", layout="wide")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent; }
header [data-testid="stToolbar"] { visibility: hidden; }

@keyframes gradientMove {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes pulse {0%,100%{box-shadow:0 0 0 0 rgba(255,138,92,.35)}50%{box-shadow:0 0 0 12px rgba(255,138,92,0)}}
@keyframes floatY {0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}

.topbar {
  display:flex; justify-content:space-between; align-items:center;
  padding:.95rem 1.5rem; border-radius:16px;
  background:rgba(255,255,255,.88); backdrop-filter:blur(14px);
  border:1px solid #e6ecf3; box-shadow:0 6px 24px rgba(15,32,39,.08);
  margin-bottom:1rem;
}
.brand { display:flex; gap:.7rem; align-items:center; font-weight:800; color:#123047; font-size:1.18rem; }
.brand .logo { font-size:1.7rem; }
.brand .tag { font-weight:500; color:#5b6b7b; font-size:.82rem; }
.status-chip { font-size:.74rem; padding:.38rem .85rem; border-radius:999px; background:#e6f9ee;
  color:#0a8f4d; border:1px solid #b5ecd0; font-weight:700; margin-left:.4rem; }
.status-chip.blue { background:#eef4ff; color:#2c5cc5; border-color:#c9daff; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f2027,#203a43 60%,#2c5364); }
.logo-glow { text-shadow: 0 0 22px rgba(255,215,111,.7); }
.nav-caption { font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; color:#8fb3c7;
  margin:1.2rem 0 .5rem .15rem; font-weight:700; }

/* ---------- NAV PILLS (visible on desktop AND phone) ---------- */
div[role="radiogroup"] { display:flex; flex-wrap:wrap; gap:.55rem; justify-content:center; }
div[role="radiogroup"] label {
  display:flex; align-items:center; padding:.72rem 1.2rem; border-radius:999px;
  background:#ffffff; border:1px solid #e6ecf3; cursor:pointer;
  transition:all .25s ease; margin:0; box-shadow:0 4px 14px rgba(15,32,39,.06);
}
div[role="radiogroup"] label:hover { transform:translateY(-2px); box-shadow:0 8px 20px rgba(15,32,39,.12); }
div[role="radiogroup"] label p { margin:0; color:#123047 !important; font-weight:600; font-size:.92rem; }
div[role="radiogroup"] input[type="radio"] { display:none; }
div[role="radiogroup"] label:has(input:checked) {
  background:linear-gradient(90deg,#ff8a5c,#ff5c8a);
  border-color:transparent; box-shadow:0 8px 22px rgba(255,92,138,.35);
}
div[role="radiogroup"] label:has(input:checked) p { color:#ffffff !important; }

.hero {
  padding:2.6rem 2.4rem; border-radius:22px;
  background:linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#6a3093);
  background-size:400% 400%; animation:gradientMove 14s ease infinite;
  color:#ffffff; margin-bottom:1.5rem;
  box-shadow:0 12px 40px rgba(15,32,39,.35);
}
.hero h1 { font-size:2.7rem; font-weight:800; margin:.4rem 0 0 0; }
.hero p { opacity:.88; margin-top:.6rem; font-size:1.05rem; max-width:900px; }
.chip { display:inline-block; padding:.38rem .95rem; border-radius:999px;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28);
  font-size:.8rem; margin-right:.45rem; margin-bottom:.5rem; animation:floatY 5s ease infinite; }
.grad-text { background:linear-gradient(90deg,#ffd76f,#ff8a5c,#ff5c8a);
  -webkit-background-clip:text; background-clip:text; color:transparent; font-weight:800; }

.card {
  background:rgba(255,255,255,.92); backdrop-filter:blur(14px);
  border:1px solid #e6ecf3; border-radius:18px;
  padding:1.25rem 1.45rem; box-shadow:0 8px 32px rgba(15,32,39,.08);
  height:100%; transition:transform .25s ease, box-shadow .25s ease;
}
.card:hover { transform:translateY(-5px); box-shadow:0 14px 40px rgba(15,32,39,.14); }
.card h3 { margin:0 0 .35rem 0; font-size:.92rem; font-weight:600; color:#5b6b7b; }
.card .value { font-size:1.75rem; font-weight:800; color:#123047; }
.card p { margin:.25rem 0 0 0; color:#5b6b7b; font-size:.86rem; }

.section-title { font-size:1.15rem; font-weight:700; color:#123047; margin:1.4rem 0 .6rem 0;
  position:relative; padding-bottom:.45rem; }
.section-title::after { content:''; position:absolute; left:0; bottom:0; width:70px; height:4px;
  border-radius:4px; background:linear-gradient(90deg,#ff8a5c,#ff5c8a); }

.badge { display:inline-block; padding:.55rem 1.3rem; border-radius:999px; font-weight:700;
  font-size:1rem; animation:pulse 2.6s infinite; }
.badge-low  { background:#e6f9ee; color:#0a8f4d; border:1px solid #b5ecd0; }
.badge-med  { background:#fff6e5; color:#b26a00; border:1px solid #ffe1ad; }
.badge-high { background:#ffe9e9; color:#c62828; border:1px solid #ffc9c9; }

.rec { background:#ffffff; border-left:4px solid #2c74b3; border-radius:12px;
  padding:.75rem 1.05rem; margin-bottom:.6rem; font-size:.95rem; color:#28405a;
  box-shadow:0 4px 16px rgba(15,32,39,.06); }

.stFormSubmitButton button, .stButton button, .stDownloadButton button {
  background:linear-gradient(90deg,#ff8a5c,#ff5c8a);
  color:#ffffff; font-weight:700; border:none; border-radius:14px;
  padding:.65rem 1.2rem; box-shadow:0 8px 22px rgba(255,92,138,.30);
  transition:transform .2s ease, box-shadow .2s ease;
}
.stFormSubmitButton button:hover, .stButton button:hover, .stDownloadButton button:hover {
  transform:translateY(-2px); box-shadow:0 12px 28px rgba(255,92,138,.42); color:#ffffff;
}

.grade-table { width:100%; border-collapse:collapse; border-radius:14px; overflow:hidden;
  box-shadow:0 6px 24px rgba(15,32,39,.08); }
.grade-table th { background:linear-gradient(90deg,#203a43,#2c5364); color:#fff;
  padding:.7rem 1rem; text-align:left; font-size:.9rem; }
.grade-table td { padding:.6rem 1rem; font-size:.9rem; color:#28405a; background:#ffffff;
  border-bottom:1px solid #eef2f6; }
.grade-table tr:nth-child(even) td { background:#f7fafc; }

.footer-bar { margin-top:2rem; padding:1rem 1.4rem; border-radius:16px; text-align:center;
  font-size:.85rem; color:#e8f0f7; background:linear-gradient(90deg,#0f2027,#203a43,#2c5364); }
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return model, columns, metrics


try:
    model, columns, metrics = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Please run `python train.py` first.")
    st.stop()

DEFAULT_MEDIANS = {
    "semester": 4.0, "current_cgpa": 7.0, "last_sgpa": 7.0, "backlogs": 1.0,
    "attendance_percent": 78.0, "internal_marks": 65.0, "assignment_avg": 70.0,
    "study_hours_per_week": 14.0, "sleep_hours_per_night": 7.0,
    "internet_hours_per_day": 3.0, "motivation_level": 3.0,
    "family_support_level": 3.0, "part_time_job": 0.0,
}
MEDIANS = metrics.get("medians", DEFAULT_MEDIANS)
RISK_COLORS = {"Low Risk": "#0a8f4d", "Medium Risk": "#e69500", "High Risk": "#c62828"}
BRANCHES = ["CSE", "IT", "AI & DS", "ECE", "EEE", "Mechanical", "Civil", "Chemical", "Biotech", "Other"]


def risk_info(score):
    if score >= 70:
        return "Low Risk", "badge-low", "#0a8f4d"
    if score >= 50:
        return "Medium Risk", "badge-med", "#b26a00"
    return "High Risk", "badge-high", "#c62828"


def grade_of(sgpa):
    if sgpa >= 9.5: return "O (Outstanding)"
    if sgpa >= 9.0: return "A+ (Excellent)"
    if sgpa >= 8.0: return "A (Very Good)"
    if sgpa >= 7.0: return "B+ (Good)"
    if sgpa >= 6.0: return "B (Above Average)"
    if sgpa >= 5.0: return "C (Average)"
    if sgpa >= 4.0: return "P (Pass)"
    return "F (Fail Risk)"


def pretty(name):
    return name.replace("_", " ").title()


def card_html(title, value, sub=""):
    return f"<div class='card'><h3>{title}</h3><div class='value'>{value}</div><p>{sub}</p></div>"


def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        font=dict(family="Poppins", size=12, color="#28405a"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def score_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        number={"suffix": " %", "font": {"size": 46, "color": "#123047"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0, 50], "color": "#ffe3e3"},
                {"range": [50, 70], "color": "#fff1d6"},
                {"range": [70, 100], "color": "#dcf5e7"},
            ],
        },
    ))
    return style_fig(fig, 320)


def local_contributions(inputs):
    base_pred = model.predict(pd.DataFrame([inputs])[columns])[0]
    rows = []
    for col in columns:
        alt = dict(inputs)
        alt[col] = MEDIANS.get(col, alt[col])
        alt_pred = model.predict(pd.DataFrame([alt])[columns])[0]
        rows.append((pretty(col), round(base_pred - alt_pred, 2)))
    rows.sort(key=lambda r: r[1])
    return pd.DataFrame(rows, columns=["factor", "impact"])


def get_recommendations(inputs, pred_sgpa):
    recs = []
    if inputs["attendance_percent"] < 75:
        recs.append("📅 Attendance is below the 75% norm — you may be barred from end-sem exams. Plan a recovery schedule with your class advisor.")
    if inputs["backlogs"] > 0:
        recs.append(f"🧾 {inputs['backlogs']} active backlog(s). Clear them in the next supplementary exam — most campus placement drives require zero backlogs.")
    if inputs["current_cgpa"] < 6.5:
        recs.append("🎯 CGPA is below the 6.5 cutoff used by many recruiters. Focus on internals and assignments this semester to pull it up.")
    if inputs["study_hours_per_week"] < 10:
        recs.append("📚 Increase structured self-study to 10–15 hrs/week with spaced revision before internals.")
    if inputs["internal_marks"] < 60:
        recs.append("🧑‍🏫 Internal marks are low — meet subject faculty for mentoring and solve previous-year question papers.")
    if inputs["assignment_avg"] < 60:
        recs.append("📝 Submit assignments and lab records on time; they carry direct internal weightage.")
    if inputs["internet_hours_per_day"] > 4:
        recs.append("📵 Cut non-academic screen time; use app/website blockers during study slots.")
    if inputs["sleep_hours_per_night"] < 6:
        recs.append("😴 Under 6 hrs of sleep hurts retention — fix your routine before internal exam weeks.")
    if inputs["motivation_level"] <= 2:
        recs.append("🧭 Talk to a counselor or mentor; set small weekly targets (one unit + one PYQ set).")
    if inputs["part_time_job"] == 1 and inputs["study_hours_per_week"] < 12:
        recs.append("⏳ Balance internship/part-time work with a fixed weekly timetable.")
    if pred_sgpa >= 8.5:
        recs.append("🚀 Strong profile — consider GATE / CAT / internship prep and research or open-source work to boost your resume.")
    if not recs:
        recs.append("✅ Profile looks balanced. Keep monitoring internals and attendance weekly.")
    return recs


SAMPLE_INPUTS = {
    "semester": 4, "current_cgpa": 7.0, "last_sgpa": 7.0, "backlogs": 0,
    "attendance_percent": 80.0, "internal_marks": 68.0, "assignment_avg": 70.0,
    "study_hours_per_week": 15.0, "sleep_hours_per_night": 7.0,
    "internet_hours_per_day": 2.5, "motivation_level": 3, "family_support_level": 3,
    "part_time_job": 0,
}

GRADE_TABLE_HTML = """
<table class='grade-table'>
<tr><th>SGPA / CGPA (10-point)</th><th>Grade</th><th>≈ Percentage (GPA × 9.5)</th></tr>
<tr><td>9.5 – 10.0</td><td>O (Outstanding)</td><td>90 – 100%</td></tr>
<tr><td>9.0 – 9.49</td><td>A+ (Excellent)</td><td>85 – 90%</td></tr>
<tr><td>8.0 – 8.99</td><td>A (Very Good)</td><td>76 – 85%</td></tr>
<tr><td>7.0 – 7.99</td><td>B+ (Good)</td><td>66 – 76%</td></tr>
<tr><td>6.0 – 6.99</td><td>B (Above Average)</td><td>57 – 66%</td></tr>
<tr><td>5.0 – 5.99</td><td>C (Average)</td><td>47 – 57%</td></tr>
<tr><td>4.0 – 4.99</td><td>P (Pass)</td><td>38 – 47%</td></tr>
<tr><td>&lt; 4.0</td><td>F (Fail)</td><td>&lt; 38%</td></tr>
</table>
"""

# ---------------- SIDEBAR (professional nav) ----------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:1.3rem 0 .6rem 0;'>"
        "<div class='logo-glow' style='font-size:2.5rem;'>🎓</div>"
        "<div style='font-size:1.45rem; font-weight:800; color:#ffffff;'>EduPredict</div>"
        "<div style='font-size:.78rem; color:#9fc0d2;'>Indian College Performance Intelligence</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nav-caption'>Model Status</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card' style='padding:.95rem 1.05rem;'>"
        "<h3>Random Forest • v2.0</h3>"
        f"<p style='margin:0; color:#5b6b7b !important; font-size:.85rem;'>"
        f"Trained on <b>{metrics.get('rows', '—')}</b> students<br>"
        f"MAE ±{metrics.get('mae', '—')} • R² {metrics.get('r2', '—')}<br>"
        "Scale: 10-point GPA • % ≈ GPA × 9.5</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nav-caption'>Session</div>", unsafe_allow_html=True)
    st.markdown("<span class='status-chip'>● Model Online</span>", unsafe_allow_html=True)

# ---------------- TOP NAVBAR (all pages) ----------------
st.markdown(
    "<div class='topbar'>"
    "<div class='brand'><span class='logo'>🎓</span> EduPredict "
    "<span class='tag'>| Indian College Performance Intelligence</span></div>"
    "<div><span class='status-chip'>● Model Online</span>"
    "<span class='status-chip blue'>v2.0</span></div>"
    "</div>",
    unsafe_allow_html=True,
)
page = st.radio(
    "Navigation",
    ["🏠 Home", "🎯 Predict Performance", "📊 Batch Analysis", "ℹ️ About & Help"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<div style='height:.8rem;'></div>", unsafe_allow_html=True)

# ================= HOME =================
if page == "🏠 Home":
    st.markdown(
        "<div class='hero'>"
        "<span class='chip'>🇮🇳 Built for Indian College Students</span>"
        "<span class='chip'>📊 CGPA / SGPA Aware</span>"
        "<span class='chip'>🧾 Backlog Alerts</span>"
        "<span class='chip'>🚨 Early Risk Detection</span>"
        "<h1>Student Performance <span class='grad-text'>Predictor</span></h1>"
        "<p>An AI-powered early-warning system for colleges — predicts percentage, SGPA and CGPA trend, "
        "and flags at-risk students before it is too late.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card_html("Students in training data", metrics.get("rows", "—"), "Records used to train the model"), unsafe_allow_html=True)
    c2.markdown(card_html("Average prediction error", f"±{metrics.get('mae', '—')}", "Percentage points (MAE)"), unsafe_allow_html=True)
    c3.markdown(card_html("Model R² score", metrics.get("r2", "—"), "How well the model explains results"), unsafe_allow_html=True)
    c4.markdown(card_html("Grading scale", "10-point", "GPA system • % ≈ GPA × 9.5"), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>What EduPredict does</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='card'><h3>🚨 Early Risk Detection</h3><p>Predicts end-sem percentage and flags high-risk students early in the semester.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='card'><h3>📊 CGPA / SGPA Insights</h3><p>Shows predicted SGPA, grade, and estimated CGPA after this semester on the 10-point scale.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='card'><h3>📦 Batch Analytics</h3><p>Upload a whole class CSV and get risk distribution charts and result downloads instantly.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>How it works</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card'><p style='font-size:.95rem; color:#28405a;'>"
        "<b>1.</b> Enter a student's semester, CGPA/SGPA, backlogs, attendance and lifestyle profile (or upload a class CSV).<br>"
        "<b>2.</b> The Random Forest model predicts the end-sem percentage out of 100.<br>"
        "<b>3.</b> EduPredict converts it to SGPA / grade, estimates CGPA trend, and shows risk + driving factors.<br>"
        "<b>4.</b> Faculty and counselors use it as a mentoring support tool — never as a final decision."
        "</p></div>",
        unsafe_allow_html=True,
    )

# ================= PREDICT =================
elif page == "🎯 Predict Performance":
    with st.form("predict_form"):
        st.markdown("<div class='section-title'>🎓 Semester & GPA Profile</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            semester = st.slider("Semester", 1, 8, 4)
        with c2:
            branch = st.selectbox("Branch", BRANCHES)
        with c3:
            current_cgpa = st.slider("Current CGPA (0-10)", 4.0, 10.0, 7.0, 0.1)
        with c4:
            last_sgpa = st.slider("Last semester SGPA", 3.5, 10.0, 7.0, 0.1)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            attendance = st.slider("Attendance %", 0.0, 100.0, 80.0, 1.0)
        with c2:
            internal_marks = st.slider("Internal marks (0-100)", 0.0, 100.0, 68.0, 1.0)
        with c3:
            assignment_avg = st.slider("Assignment avg (0-100)", 0.0, 100.0, 70.0, 1.0)
        with c4:
            backlogs = st.number_input("Active backlogs", 0, 10, 0, 1)

        st.markdown("<div class='section-title'>⏰ Study & Lifestyle</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            study_hours = st.number_input("Study hours / week", 0.0, 60.0, 15.0, 0.5)
        with c2:
            sleep_hours = st.slider("Sleep hours / night", 3.0, 12.0, 7.0, 0.5)
        with c3:
            internet_hours = st.slider("Social media hrs / day", 0.0, 12.0, 2.5, 0.5)

        st.markdown("<div class='section-title'>🧑‍ Personal Factors</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            motivation = st.select_slider("Motivation (1-5)", options=[1, 2, 3, 4, 5], value=3)
        with c2:
            family_support = st.select_slider("Family support (1-5)", options=[1, 2, 3, 4, 5], value=3)
        with c3:
            st.write("")
            part_time_job = st.checkbox("Internship / part-time job", value=False)

        submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True)

    if submitted:
        inputs = {
            "semester": int(semester),
            "current_cgpa": float(current_cgpa),
            "last_sgpa": float(last_sgpa),
            "backlogs": int(backlogs),
            "attendance_percent": float(attendance),
            "internal_marks": float(internal_marks),
            "assignment_avg": float(assignment_avg),
            "study_hours_per_week": float(study_hours),
            "sleep_hours_per_night": float(sleep_hours),
            "internet_hours_per_day": float(internet_hours),
            "motivation_level": int(motivation),
            "family_support_level": int(family_support),
            "part_time_job": 1 if part_time_job else 0,
        }

        input_df = pd.DataFrame([inputs])[columns]
        pred = float(model.predict(input_df)[0])
        label, badge_class, color = risk_info(pred)
        pred_sgpa = round(min(10.0, pred / 9.5), 2)
        if semester <= 1:
            est_cgpa = pred_sgpa
        else:
            est_cgpa = round((current_cgpa * (semester - 1) + pred_sgpa) / semester, 2)

        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.plotly_chart(score_gauge(pred, color), use_container_width=True)
        with c2:
            st.markdown(f"<span class='badge {badge_class}'>{label}</span>", unsafe_allow_html=True)
            st.write("")
            ca, cb = st.columns(2)
            ca.markdown(card_html("Predicted %", f"{pred:.1f}", "End-sem percentage"), unsafe_allow_html=True)
            cb.markdown(card_html("Predicted SGPA", f"{pred_sgpa:.2f}", f"Grade: {grade_of(pred_sgpa)}"), unsafe_allow_html=True)
            cc, cd = st.columns(2)
            cc.markdown(card_html("Est. CGPA after sem", f"{est_cgpa:.2f}", f"Semester {semester} • {branch}"), unsafe_allow_html=True)
            cd.markdown(card_html("Likely zone", "Strong Pass" if pred >= 70 else ("Pass" if pred >= 50 else "Fail Risk"), "Cut-offs: 50 / 70"), unsafe_allow_html=True)

        if backlogs > 0:
            st.markdown(f"<div class='rec'>🧾 <b>Backlog alert:</b> {backlogs} active backlog(s). Most placement drives and higher-study applications require zero backlogs — plan supplementary exam prep now.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>🧠 What is driving this prediction?</div>", unsafe_allow_html=True)
        contrib = local_contributions(inputs)
        fig_bar = go.Figure(go.Bar(
            x=contrib["impact"], y=contrib["factor"], orientation="h",
            marker_color=["#0a8f4d" if v >= 0 else "#c62828" for v in contrib["impact"]],
        ))
        fig_bar.update_layout(xaxis_title="Impact on predicted % (vs. average student)")
        st.plotly_chart(style_fig(fig_bar, 400), use_container_width=True)
        st.caption("Green = factor pushes the score above an average profile. Red = it pulls the score down.")

        st.markdown("<div class='section-title'>💡 Recommended Actions</div>", unsafe_allow_html=True)
        for rec in get_recommendations(inputs, pred_sgpa):
            st.markdown(f"<div class='rec'>{rec}</div>", unsafe_allow_html=True)

        st.caption("SGPA ≈ Percentage ÷ 9.5 (common Indian university conversion). This is an estimate for academic support only.")

# ================= BATCH =================
elif page == "📊 Batch Analysis":
    st.markdown("<div class='section-title'>Batch prediction using CSV</div>", unsafe_allow_html=True)
    st.markdown("Upload a class CSV to analyse many students at once. Do not include names, roll numbers or contact details in public demos.")

    template_df = pd.DataFrame([SAMPLE_INPUTS])[columns]
    st.download_button("⬇️ Download CSV template", template_df.to_csv(index=False).encode("utf-8"), "student_input_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload class CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            missing_columns = [c for c in columns if c not in df.columns]
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
            else:
                for c in columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=columns)
                if df.empty:
                    st.warning("No valid rows found after cleaning the CSV.")
                else:
                    df["predicted_percentage"] = model.predict(df[columns]).round(1)
                    df["predicted_sgpa"] = (df["predicted_percentage"] / 9.5).clip(upper=10).round(2)
                    df["grade"] = df["predicted_sgpa"].apply(grade_of)
                    df["risk_level"] = df["predicted_percentage"].apply(lambda s: risk_info(s)[0])

                    st.success(f"Batch prediction completed for {len(df)} students.")
                    st.dataframe(df, use_container_width=True)

                    c1, c2 = st.columns(2)
                    counts = df["risk_level"].value_counts()
                    fig_pie = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                                               marker_colors=[RISK_COLORS.get(k, "#888") for k in counts.index]))
                    fig_pie.update_layout(title="Risk distribution")
                    c1.plotly_chart(style_fig(fig_pie, 360), use_container_width=True)

                    fig_hist = go.Figure(go.Histogram(x=df["predicted_percentage"], nbinsx=15, marker_color="#2c74b3"))
                    fig_hist.update_layout(title="Predicted % spread")
                    c2.plotly_chart(style_fig(fig_hist, 360), use_container_width=True)

                    st.download_button("⬇️ Download results", df.to_csv(index=False).encode("utf-8"), "predicted_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process CSV file: {e}")

# ================= ABOUT =================
elif page == "ℹ️ About & Help":
    st.markdown("<div class='section-title'>Model card</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card'><p style='color:#28405a;'>"
        "<b>Algorithm:</b> Random Forest Regressor (scikit-learn)<br>"
        f"<b>Training rows:</b> {metrics.get('rows', '—')} &nbsp;|&nbsp; <b>MAE:</b> ±{metrics.get('mae', '—')} &nbsp;|&nbsp; <b>R²:</b> {metrics.get('r2', '—')}<br>"
        "<b>Target:</b> End-semester percentage (0–100)<br>"
        "<b>GPA scale:</b> 10-point • Percentage ≈ GPA × 9.5</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Indian grading reference (10-point scale)</div>", unsafe_allow_html=True)
    st.markdown(GRADE_TABLE_HTML, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Required CSV columns</div>", unsafe_allow_html=True)
    st.code(", ".join(columns))

    st.markdown("<div class='section-title'>Privacy & ethics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rec'>🔒 Do not upload names, roll numbers, emails or phone numbers in public demos.</div>"
        "<div class='rec'>⚖️ Predictions support mentoring decisions — they never replace human judgment.</div>"
        "<div class='rec'>🔄 Retrain each semester with fresh anonymised data: replace data/students.csv and run python train.py.</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='footer-bar'>EduPredict v2.0 • Made with ❤️ for Indian college students • Predictions support mentoring, not final decisions.</div>", unsafe_allow_html=True)