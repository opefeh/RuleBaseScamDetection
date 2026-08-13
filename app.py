"""
app.py — Rule-Based Scam Detection System for WhatsApp and SMS Messages in Nigeria
MSc Project Demonstration — Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import io

from model import detect_scam, detect_scam_dict

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigeria WhatsApp/SMS Scam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stSidebar"] { background: #1a2744; }
[data-testid="stSidebar"] * { color: #e8edf5 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; }

/* ── Header banner ── */
.main-header {
    background: linear-gradient(135deg, #1a2744 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 24px;
    color: #fff;
    text-align: center;
    box-shadow: 0 6px 32px rgba(37,99,235,.25);
}
.main-header h1 { font-size: 2rem; font-weight: 800; margin: 0 0 6px; }
.main-header p  { font-size: 1rem; color: #bfdbfe; margin: 0; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,.15);
    border-radius: 20px;
    padding: 4px 16px;
    font-size: .78rem;
    margin-top: 12px;
    letter-spacing: .5px;
}

/* ── Result cards ── */
.result-card {
    border-radius: 14px;
    padding: 24px 28px;
    margin: 16px 0;
    box-shadow: 0 4px 18px rgba(0,0,0,.08);
}
.scam-card    { background: #fff5f5; border-left: 6px solid #e53e3e; }
.suspicious-card { background: #fffbeb; border-left: 6px solid #d69e2e; }
.legitimate-card { background: #f0fff4; border-left: 6px solid #38a169; }

/* ── Score gauge ── */
.score-pill {
    display: inline-block;
    border-radius: 50px;
    padding: 8px 28px;
    font-size: 1.6rem;
    font-weight: 800;
    color: #fff;
    margin: 10px 0;
}
.score-high   { background: linear-gradient(90deg,#e53e3e,#c53030); }
.score-medium { background: linear-gradient(90deg,#d69e2e,#b7791f); }
.score-low    { background: linear-gradient(90deg,#38a169,#276749); }

/* ── Indicator tags ── */
.ind-tag {
    display: inline-block;
    background: #ebf4ff;
    color: #2b6cb0;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px 4px;
    font-size: .82rem;
    font-weight: 600;
    border: 1px solid #bee3f8;
}

/* ── Sample message buttons ── */
.sample-btn { font-size: .82rem !important; }

/* ── Warning banner ── */
.warn-box {
    border-radius: 12px;
    padding: 16px 22px;
    margin: 12px 0;
    font-weight: 600;
    font-size: .95rem;
}
.warn-scam    { background:#fed7d7; color:#c53030; border:1px solid #fc8181; }
.warn-sus     { background:#fefdeb; color:#744210; border:1px solid #f6e05e; }
.warn-legit   { background:#c6f6d5; color:#22543d; border:1px solid #68d391; }

/* ── Metric tiles ── */
.metric-tile {
    background:#fff;
    border-radius:12px;
    padding:18px 22px;
    text-align:center;
    box-shadow:0 2px 12px rgba(0,0,0,.07);
}
.metric-tile .val { font-size:1.9rem; font-weight:800; }
.metric-tile .lbl { font-size:.82rem; color:#718096; margin-top:2px; }

/* ── Footer ── */
.footer {
    text-align:center;
    color:#718096;
    font-size:.8rem;
    padding:30px 0 10px;
    border-top:1px solid #e2e8f0;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Navigation")
    page = st.radio(
        "",
        ["🔍 Analyse a Message", "📊 Dataset Evaluation", "ℹ️ About the System"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This tool uses a **rule-based scoring engine** with 20 indicator "
        "categories to classify WhatsApp and SMS messages as:\n"
        "- 🔴 **Scam**\n- 🟡 **Suspicious**\n- 🟢 **Legitimate**"
    )
    st.markdown("---")
    st.markdown("**Score thresholds**")
    st.markdown("- 0–29 → Legitimate / Low\n- 30–54 → Suspicious / Medium\n- 55–100 → Scam / High")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🛡️ Nigeria WhatsApp & SMS Scam Detection System</h1>
  <p>Rule-Based Prototype · Explainable Classification · Nigerian Scam Patterns</p>
  <span class="badge">Prototype v1.0</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analyse a Message
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔍 Analyse a Message":

    # ── Sample messages ──────────────────────────────────────────────────────
    SAMPLES = {
        "🔴 Investment Scam": (
            "Investment opportunity: invest ₦100,000 today and receive triple payment "
            "in 24 hours. Limited slot available."
        ),
        "🔴 OTP Phishing": (
            "Dear customer, your BVN has been blocked. Send your OTP now to reactivate "
            "your account immediately."
        ),
        "🔴 Fake Job + Fee": (
            "N-Power recruitment is ongoing. Pay ₦3,000 form fee to secure your interview slot."
        ),
        "🟡 Telecom Notice": (
            "9mobile: Your SIM registration status requires attention. Visit the nearest "
            "official service centre with valid ID."
        ),
        "🟡 Loan Inquiry": (
            "This loan agent said I should pay a small fee first. Does it sound real?"
        ),
        "🟢 Work Message": (
            "Please send the updated document before 6pm so I can review it."
        ),
        "🟢 Personal": (
            "Good morning, please are we still meeting by 4pm today?"
        ),
    }

    st.markdown("### 💬 Enter or Paste a Message")

    # Quick-fill buttons
    st.markdown("**Try a sample message:**")
    cols = st.columns(len(SAMPLES))
    selected_sample = ""
    for i, (label, text) in enumerate(SAMPLES.items()):
        if cols[i].button(label, key=f"s{i}", use_container_width=True):
            selected_sample = text

    # Text area
    default_text = selected_sample if selected_sample else ""
    message_input = st.text_area(
        "Message Text",
        value=default_text,
        height=150,
        placeholder="Paste or type a WhatsApp or SMS message here …",
        label_visibility="collapsed",
    )

    col_btn, col_clr = st.columns([3, 1])
    analyse_clicked = col_btn.button("🔍 Analyse Message", type="primary", use_container_width=True)
    if col_clr.button("🗑️ Clear", use_container_width=True):
        message_input = ""

    # ── Analysis results ─────────────────────────────────────────────────────
    if analyse_clicked and message_input.strip():
        result = detect_scam(message_input.strip())

        label     = result.predicted_label
        score     = result.risk_score
        level     = result.risk_level
        indicators = result.detected_indicators
        expl      = result.explanation

        # Colour mappings
        card_cls  = {"Scam":"scam","Suspicious":"suspicious","Legitimate":"legitimate"}[label]
        score_cls = {"High":"score-high","Medium":"score-medium","Low":"score-low"}[level]
        emoji     = {"Scam":"🔴","Suspicious":"🟡","Legitimate":"🟢"}[label]
        warn_cls  = {"Scam":"warn-scam","Suspicious":"warn-sus","Legitimate":"warn-legit"}[label]
        warn_msg  = {
            "Scam": "⚠️ WARNING: This message shows strong signs of a scam. Do NOT click any links, share personal details, or send money. Report and delete immediately.",
            "Suspicious": "⚠️ CAUTION: This message has suspicious elements. Verify the sender through official channels before taking any action.",
            "Legitimate": "✅ This message appears legitimate. No significant scam indicators were detected.",
        }[label]

        st.markdown(f"""
        <div class="result-card {card_cls}-card">
          <h3>{emoji} Classification: <strong>{label}</strong></h3>
          <div class="score-pill {score_cls}">Risk Score: {score} / 100</div>
          &nbsp;&nbsp;<span style="font-size:1.05rem;font-weight:700;">Risk Level: {level}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="warn-box {warn_cls}">{warn_msg}</div>', unsafe_allow_html=True)

        # Progress bar
        st.markdown("**Risk Score Gauge**")
        colour = "#e53e3e" if score >= 55 else ("#d69e2e" if score >= 30 else "#38a169")
        st.progress(score / 100)

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🚩 Detected Indicators")
            if indicators:
                tags = "".join(f'<span class="ind-tag">{ind}</span>' for ind in indicators)
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.markdown("✅ No scam indicators detected.")

        with col_right:
            st.markdown("#### 📋 Explanation")
            st.info(expl)

    elif analyse_clicked:
        st.warning("Please enter a message to analyse.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Dataset Evaluation
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Evaluation":

    st.markdown("### 📊 Dataset Evaluation")
    st.markdown(
        "Upload the dataset Excel file to evaluate the rule-based model against the "
        "expected labels and see full performance metrics."
    )

    uploaded = st.file_uploader(
        "Upload dataset (.xlsx)",
        type=["xlsx"],
        help="Upload nigeria_whatsapp_sms_scam_detection_dataset.xlsx",
    )

    if uploaded:
        with st.spinner("Running rule-based model on all messages …"):
            df = pd.read_excel(uploaded)
            results = df['Message_Text'].apply(detect_scam_dict).apply(pd.Series)
            df = pd.concat([df, results], axis=1)
            df['Correct_Classification'] = (
                df['Expected_Label'] == df['System_Prediction']
            ).map({True: "✅ Correct", False: "❌ Incorrect"})

            y_true = df['Expected_Label']
            y_pred = df['System_Prediction']
            labels = ['Legitimate', 'Suspicious', 'Scam']

            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            cm   = confusion_matrix(y_true, y_pred, labels=labels)
            cr   = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # ── Top metrics ──
        st.markdown("#### Overall Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl, colour in [
            (m1, acc,  "Accuracy",            "#2563eb"),
            (m2, prec, "Precision (weighted)", "#7c3aed"),
            (m3, rec,  "Recall (weighted)",    "#059669"),
            (m4, f1,   "F1 Score (weighted)",  "#d97706"),
        ]:
            col.markdown(f"""
            <div class="metric-tile">
              <div class="val" style="color:{colour}">{val:.1%}</div>
              <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Confusion matrix ──
        col_cm, col_cr = st.columns(2)

        with col_cm:
            st.markdown("#### Confusion Matrix")
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            cm_df.index.name = "Actual \\ Predicted"
            st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)
            st.caption(
                "Rows = actual label · Columns = predicted label · "
                "Diagonal = correct classifications"
            )

        with col_cr:
            st.markdown("#### Per-Class Metrics")
            cr_df = pd.DataFrame(cr).transpose()
            cr_df = cr_df.loc[labels + ['accuracy', 'macro avg', 'weighted avg']]
            st.dataframe(
                cr_df.style.format("{:.3f}").background_gradient(cmap="Greens", subset=['f1-score']),
                use_container_width=True,
            )

        st.markdown("---")

        # ── Dataset distribution ──
        st.markdown("#### Label Distribution")
        dist_col1, dist_col2, dist_col3 = st.columns(3)
        for col, lbl, clr in [
            (dist_col1, "Expected_Label",    "#2563eb"),
            (dist_col2, "System_Prediction", "#7c3aed"),
            (dist_col3, "System_Risk_Level", "#059669"),
        ]:
            col.markdown(f"**{lbl}**")
            col.dataframe(df[lbl].value_counts().rename_axis(lbl).reset_index(name="Count"),
                          use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Prediction preview ──
        st.markdown("#### Prediction Preview (first 50 rows)")
        preview_cols = [
            'Message_ID', 'Message_Type', 'Message_Text',
            'Expected_Label', 'System_Prediction',
            'System_Risk_Score', 'System_Risk_Level',
            'Correct_Classification',
        ]
        st.dataframe(df[preview_cols].head(50), use_container_width=True, height=380)

        # ── Download ──
        st.markdown("---")
        st.markdown("#### Download Full Results")
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        st.download_button(
            "⬇️ Download Evaluation Results (.xlsx)",
            data=output.getvalue(),
            file_name="scam_detection_evaluation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    else:
        st.info("📂 Upload the dataset file above to begin evaluation.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — About
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About the System":

    st.markdown("### ℹ️ About the System")

    st.markdown("""
    #### Project Overview
    This application is a prototype developed as part of an MSc Computer Science research project titled:

    > **"Design and Implementation of a Rule-Based Scam Detection System for WhatsApp and SMS Messages in Nigeria"**

    #### Detection Methodology
    The system uses a **rule-based, weighted scoring engine** — not machine learning. Each incoming message
    is scanned against **20 indicator categories** derived from real-world Nigerian scam patterns.
    Each detected indicator contributes a weighted score to a cumulative risk total.

    #### Rule Categories
    """)

    rules_info = [
        ("OTP/PIN/Password/CVV Request", 30, "Requests for authentication credentials"),
        ("BVN/Account/Card Detail Request", 30, "Requests for financial identifiers"),
        ("Suspicious/Shortened URL", 25, "Short links or phishing-style URLs"),
        ("Financial Request/Processing Fee", 25, "Upfront payments demanded"),
        ("Personal Data Harvesting", 25, "NIN, DOB, or identity information requests"),
        ("Government Programme Fee Scam", 25, "N-Power/grant fee demands"),
        ("Bank/Government/Telecom Impersonation", 20, "False authority claims"),
        ("Account Blocked/Suspended Threat", 20, "Fear-based account threats"),
        ("Emergency Money Request", 20, "Crisis fabrication for money"),
        ("SIM/NIN Update Threat", 20, "SIM deactivation or NIN threats"),
        ("Suspicious External Domain Link", 20, "Phishing domain patterns"),
        ("Advance-Fee/Classic Scam Phrase", 20, "419-style language"),
        ("Fake Reward/Prize/Grant/Lottery", 15, "Unsolicited prize claims"),
        ("Fake Job/Recruitment Offer", 15, "Unrealistic job advertisements"),
        ("Fake Loan Offer", 15, "Instant or approved loan promises"),
        ("Investment Scam/Doubling Scheme", 15, "Guaranteed high-return investments"),
        ("Investment + Urgent Combo", 20, "Investment + extreme urgency"),
        ("Urgent/Pressure Language", 15, "Artificial deadline pressure"),
        ("Suspicious Link in Forwarded Message", 15, "Forwarded phishing links"),
        ("Vague Account Update Request", 15, "Generic account update pretexts"),
    ]

    rules_df = pd.DataFrame(rules_info, columns=["Rule / Indicator", "Score Weight", "Description"])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    st.markdown("""
    #### Classification Thresholds
    | Score Range | Label | Risk Level |
    |-------------|-------|------------|
    | 0 – 29 | Legitimate | Low |
    | 30 – 54 | Suspicious | Medium |
    | 55 – 100 | Scam | High |

    #### Limitations
    - The system relies on pattern matching and may miss novel phrasing.
    - Context beyond the message text (sender identity, history) is not considered.
    - Legitimate messages that happen to contain trigger words may be flagged incorrectly.
    - The system is designed for the Nigerian context and may require adaptation for other regions.

    #### Technologies Used
    - **Python 3** — core language
    - **Regular Expressions** — pattern matching engine
    - **Streamlit** — web application framework
    - **scikit-learn** — evaluation metrics
    - **pandas / openpyxl** — data handling and Excel export
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🛡️ Nigeria WhatsApp & SMS Scam Detection System &nbsp;|&nbsp;
  Rule-Based Prototype for MSc Research &nbsp;|&nbsp;
  <strong>Not intended for production deployment without further validation</strong><br>
  Built with Python · Streamlit · scikit-learn &nbsp;|&nbsp; © 2025
</div>
""", unsafe_allow_html=True)
