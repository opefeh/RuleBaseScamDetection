"""
model.py
Rule-Based Scam Detection Model for WhatsApp and SMS Messages in Nigeria
MSc Project — Design and Implementation of a Rule-Based Scam Detection System
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Data structure for a single detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    predicted_label: str
    risk_score: int
    risk_level: str
    detected_indicators: List[str]
    explanation: str


# ---------------------------------------------------------------------------
# Rule definitions — each rule is (name, pattern_list, score, explanation)
# ---------------------------------------------------------------------------

# Compile a helper: case-insensitive OR-match over a list of patterns
def _any(patterns: List[str]) -> re.Pattern:
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


# ── Rule catalogue ──────────────────────────────────────────────────────────

RULES: List[Tuple[str, re.Pattern, int, str]] = [

    # ── 1. OTP / PIN / Password / CVV request (score 30) ───────────────────
    (
        "OTP/PIN/Password/CVV Request",
        _any([
            r"\botp\b", r"one[- ]time[- ]pass(?:word|code)?",
            r"\bpin\b", r"\bpassword\b", r"\bcvv\b",
            r"send\s+(your\s+)?(pin|otp|password|cvv|code)",
            r"share\s+(your\s+)?(pin|otp|password|cvv|code)",
            r"provide\s+(your\s+)?(pin|otp|password|cvv|code)",
            r"enter\s+(your\s+)?(pin|otp|password|cvv|code)",
            r"reply\s+with\s+(your\s+)?(pin|otp|password|cvv|code)",
        ]),
        30,
        "The message requests an OTP, PIN, password, or CVV — legitimate organisations never ask for these via message.",
    ),

    # ── 2. BVN / Account number / Card detail request (score 30) ───────────
    (
        "BVN/Account/Card Detail Request",
        _any([
            r"\bbvn\b", r"bank\s+verification\s+number",
            r"account\s+number", r"card\s+(number|detail|expiry|expire)",
            r"sort\s+code", r"nuban",
            r"send\s+(your\s+)?(bvn|account|card)",
            r"provide\s+(your\s+)?(bvn|account|card)",
            r"reply\s+with\s+(your\s+)?(bvn|account|card)",
        ]),
        30,
        "The message asks for sensitive financial identifiers (BVN, account number, or card details).",
    ),

    # ── 3. Suspicious / shortened / phishing URL (score 25) ────────────────
    (
        "Suspicious/Shortened URL",
        _any([
            r"bit\.ly", r"tinyurl\.com", r"ow\.ly", r"t\.co",
            r"goo\.gl", r"is\.gd", r"cutt\.ly", r"rb\.gy",
            r"shorturl\.at", r"clck\.ru",
            r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # bare IP link
            r"click\s+(here|this\s+link|below)\s+to\s+(verify|claim|activate|confirm|unlock)",
            r"verify\s+(your\s+account|your\s+details?)\s+at\s+http",
            r"login\s+at\s+http",
        ]),
        25,
        "The message contains a shortened URL or a link designed to redirect to a phishing page.",
    ),

    # ── 4. Financial request / processing fee / registration fee (score 25) ─
    (
        "Financial Request/Processing Fee",
        _any([
            r"pay\s+(a\s+)?(small\s+)?(fee|charge|amount)",
            r"processing\s+fee", r"registration\s+fee", r"activation\s+fee",
            r"transfer\s+(a\s+)?(fee|charge|amount)",
            r"send\s+money", r"send\s+cash", r"wire\s+transfer",
            r"upfront\s+(payment|fee|charge)",
            r"before\s+(we\s+)?(release|disburse|transfer|send)\s+your",
            r"insurance\s+fee", r"clearance\s+fee", r"delivery\s+fee\s+to\s+claim",
        ]),
        25,
        "The message demands an upfront payment, processing fee, or registration fee before releasing funds or prizes.",
    ),

    # ── 5. Bank / government / telecom impersonation (score 20) ────────────
    (
        "Bank/Government/Telecom Impersonation",
        _any([
            r"\baccess\s+bank\b", r"\bgtb?\b", r"\bgtbank\b", r"\bzenith\s+bank\b",
            r"\bfirst\s+bank\b", r"\beco\s*bank\b", r"\bpolaris\s+bank\b",
            r"\bsterling\s+bank\b", r"\bfidelity\s+bank\b", r"\bunion\s+bank\b",
            r"\bubani\b", r"\bkuda\b", r"\bopay\b", r"\bpalmpay\b", r"\bmoniepoint\b",
            r"\bcbn\b", r"central\s+bank\s+of\s+nigeria",
            r"\befcc\b", r"\bndlea\b", r"\bnass\b", r"\bnimc\b", r"\bncc\b",
            r"\bnipost\b", r"\bfirs\b",
            r"\bmtn\b", r"\bairtel\b", r"\b9mobile\b", r"\bglo\b",
            r"customer\s+(care|service|support)\s+(agent|representative|officer)",
            r"(from|on\s+behalf\s+of)\s+(your\s+)?(bank|government|cbn|efcc)",
            r"official\s+(bank|government|telecom)\s+(message|notice|alert)",
        ]),
        20,
        "The message impersonates a Nigerian bank, government agency, or telecom provider to establish false trust.",
    ),

    # ── 6. Account blocked / suspended / restricted threat (score 20) ───────
    (
        "Account Blocked/Suspended/Restricted Threat",
        _any([
            r"account\s+(has\s+been\s+|is\s+|will\s+be\s+)?(blocked|suspended|restricted|frozen|deactivated|disabled)",
            r"card\s+(has\s+been\s+|is\s+|will\s+be\s+)?(blocked|suspended|restricted|frozen)",
            r"bvn\s+(has\s+been\s+|is\s+)?(blocked|suspended|flagged)",
            r"your\s+(account|card|sim|number)\s+will\s+be\s+(block|suspend|deactivat|disabled)",
            r"account\s+deactivation",
            r"(failure\s+to\s+|if\s+you\s+don.?t\s+).{0,30}(block|suspend|deactivat|restrict)",
            r"lose\s+access\s+to\s+your\s+account",
        ]),
        20,
        "The message threatens account suspension, restriction, or deactivation to create fear and prompt hasty action.",
    ),

    # ── 7. Fake reward / prize / lottery / grant (score 15) ─────────────────
    (
        "Fake Reward/Prize/Grant/Lottery",
        _any([
            r"\bwinner\b", r"you\s+have\s+won", r"congratulations.{0,30}(won|prize|award|reward)",
            r"selected.{0,30}(winner|lucky|beneficiary)",
            r"cash\s+prize", r"gift\s+(voucher|card|hamper)",
            r"lottery\s+(winner|prize|jackpot)",
            r"\bgrant\b.{0,30}(awarded|approved|selected|claim)",
            r"you\s+are\s+(a\s+)?(beneficiary|recipient)\s+of",
            r"government\s+(relief|palliative|grant|empowerment)\s+(fund|money|cash)",
            r"claim\s+your\s+(prize|reward|cash|gift|winnings)",
            r"free\s+(iphone|laptop|data|airtime|money|cash)",
        ]),
        15,
        "The message makes an unsolicited prize, reward, or grant claim — a classic advance-fee fraud tactic.",
    ),

    # ── 8. Fake job / recruitment scam (score 15) ───────────────────────────
    (
        "Fake Job/Recruitment Offer",
        _any([
            r"earn\s+(up\s+to\s+)?₦?\d[\d,]*\s*(per\s+(day|week|month)|daily|weekly|monthly)",
            r"work\s+from\s+home.{0,30}earn",
            r"no\s+experience\s+(required|needed)",
            r"recruitment\s+(exercise|portal|form).{0,20}(register|apply|pay|fee)",
            r"we\s+are\s+recruiting",
            r"vacancy.{0,30}(apply|fee|register|pay)",
            r"job\s+(offer|opportunity).{0,30}(fee|register|pay|form)",
            r"guaranteed\s+(income|salary|payment)",
        ]),
        15,
        "The message advertises an unrealistic job offer, often combined with a registration fee demand.",
    ),

    # ── 9. Fake loan offer (score 15) ───────────────────────────────────────
    (
        "Fake Loan Offer",
        _any([
            r"loan\s+(of\s+)?₦?\d[\d,]*\s+(has\s+been\s+)?approved",
            r"instant\s+loan", r"quick\s+loan", r"emergency\s+loan",
            r"loan\s+without\s+(collateral|guarantor|stress)",
            r"apply\s+for\s+(a\s+)?loan.{0,20}(today|now|instantly)",
            r"low\s+interest\s+loan",
            r"loan\s+disbursement.{0,20}(fee|register|pay)",
        ]),
        15,
        "The message promises an instant or approved loan, often accompanied by a processing fee request.",
    ),

    # ── 10. Investment / Ponzi / doubling scheme (score 15) ─────────────────
    (
        "Investment Scam/Doubling Scheme",
        _any([
            r"invest\s+(₦?\d[\d,]*).{0,30}(double|triple|get back|receive)",
            r"double\s+your\s+(money|investment|capital)",
            r"(\d+)%\s+(returns?|profit|interest)\s+(daily|weekly|monthly|guaranteed)",
            r"multiply\s+your\s+(money|investment)",
            r"crypto.{0,20}(invest|profit|return|guarantee)",
            r"forex.{0,20}(invest|profit|return|guarantee)",
            r"ponzi|pyramid\s+scheme",
            r"guaranteed\s+(profit|return|income)",
            r"risk[\s-]free\s+investment",
        ]),
        15,
        "The message promises guaranteed high returns on investment — characteristic of Ponzi or advance-fee investment fraud.",
    ),

    # ── 11. Emergency money request (score 20) ──────────────────────────────
    (
        "Emergency Money Request",
        _any([
            r"(please|kindly).{0,20}(send|transfer|lend)\s+me.{0,30}(money|cash|₦)",
            r"i\s+am\s+(in|stranded|stuck|trapped).{0,30}(please|help|send)",
            r"(accident|hospital|emergency|urgent).{0,30}(need|send|transfer)\s+(money|₦|cash)",
            r"stranded.{0,30}(money|₦|cash)",
            r"died?.{0,30}(send|transfer|help).{0,30}(money|₦)",
            r"(robbed|stolen|lost).{0,30}(money|₦|cash).{0,30}(send|transfer|help)",
        ]),
        20,
        "The message fabricates an emergency to solicit money, a common social-engineering manipulation.",
    ),

    # ── 12. SIM / NIN swap/update threat (score 20) ─────────────────────────
    (
        "SIM/NIN Update Threat",
        _any([
            r"sim\s+(swap|block|barr|deactivat|replac|updat|register)",
            r"nin\s+(link|updat|register|verif|block)",
            r"your\s+sim\s+will\s+be\s+(block|deactivat|barr)",
            r"link\s+your\s+nin",
            r"nin.{0,30}(deadline|expire|cut.?off)",
            r"sim\s+card.{0,20}(block|deactivat|suspend).{0,20}(today|now|immediately)",
        ]),
        20,
        "The message threatens SIM deactivation or demands NIN linkage to steal personal data.",
    ),

    # ── 13. Urgent / pressure language (score 15) ───────────────────────────
    (
        "Urgent/Pressure Language",
        _any([
            r"\bimmediately\b", r"\burgently\b", r"\basap\b",
            r"within\s+\d+\s+(minute|hour|day)",
            r"expires?\s+(in|today|now|soon)",
            r"limited\s+(time|slot|offer|space)",
            r"act\s+now", r"do\s+not\s+delay", r"last\s+chance",
            r"(today|now)\s+or\s+(lose|forfeit|miss)",
            r"deadline.{0,20}(today|now|hours?)",
            r"once.{0,20}(expires?|gone|over)\s+(it.?s|you)",
        ]),
        15,
        "The message uses artificial urgency or pressure to prevent the recipient from thinking critically.",
    ),

    # ── 14. Sensitive data harvesting (NIN, date of birth, mother's maiden) ──
    (
        "Personal Data Harvesting",
        _any([
            r"\bnin\b", r"national\s+identification\s+number",
            r"date\s+of\s+birth", r"\bdob\b",
            r"mother.?s\s+maiden\s+name",
            r"next\s+of\s+kin.{0,20}(detail|name|contact)",
            r"home\s+address.{0,20}(send|provide|reply)",
            r"passport\s+number.{0,20}(send|provide|reply)",
        ]),
        25,
        "The message attempts to harvest personally identifiable information that can be used for identity theft.",
    ),

    # ── 15. Advance-fee / Nigerian scam classic phrases (score 20) ──────────
    (
        "Advance-Fee/Classic Scam Phrase",
        _any([
            r"419", r"advance\s+fee",
            r"confidential\s+(business|proposal|deal)",
            r"i\s+am\s+(a\s+)?(barrister|solicitor|prince|princess|general|minister).{0,40}(million|fund|inheritance|estate)",
            r"foreign\s+(fund|transfer|inheritance|estate)",
            r"unclaimed\s+(fund|inheritance|estate|balance)",
            r"next\s+of\s+kin.{0,30}(fund|inherit|claim|million)",
            r"keep\s+this\s+(message\s+)?(confidential|secret|between\s+us)",
            r"do\s+not\s+(tell|inform|share).{0,20}(anyone|anybody|others?)",
        ]),
        20,
        "The message contains language consistent with classic Nigerian advance-fee fraud (419) schemes.",
    ),

    # ── 16. N-Power / government form fee (score 25) ────────────────────────
    (
        "Government Programme Fee Scam",
        _any([
            r"n[\s-]?power.{0,40}(fee|pay|form|register|slot)",
            r"(form|application)\s+fee.{0,30}(slot|interview|position|vacancy)",
            r"pay\s+₦?\d[\d,]*\s+(form|registration|application)\s+fee",
            r"npower\s+(recruitment|application|form)",
            r"government\s+(empowerment|programme|program).{0,30}(pay|fee|register)",
            r"federal\s+government.{0,30}(pay|fee|form|register)",
        ]),
        25,
        "The message demands a fee for a government programme (e.g., N-Power) — legitimate government programmes never charge applicants.",
    ),

    # ── 17. Suspicious external domain links (score 20) ─────────────────────
    (
        "Suspicious External Domain Link",
        _any([
            r"https?://[a-z0-9\-]+\.(example\.(org|com|net)|xyz|top|click|pw|loan|work|online|site)",
            r"support[\s-]login\.",
            r"secure[\s-]update\.",
            r"verify[\s-]ng\.",
            r"claim[\s-]ng\.",
            r"account[\s-](verify|update|secure)\.",
            r"(click|open|visit)\s+(https?://|www\.)[^\s]{5,}\.(org|net|com|xyz)\b",
        ]),
        20,
        "The message contains a suspicious external link designed to mimic a legitimate domain for phishing.",
    ),

    # ── 18. Investment + time pressure combo (score 20 extra) ───────────────
    (
        "Investment + Urgent Combo",
        _any([
            r"invest.{0,60}(24\s*hours?|today|now|immediately|limited\s+slot)",
            r"triple\s+(your\s+)?(money|investment|payment)",
            r"double\s+(your\s+)?(money|investment|payment)",
            r"receive\s+(triple|double|3x|2x).{0,20}(payment|return|profit)",
        ]),
        20,
        "The message combines an investment promise with extreme urgency or unrealistic multiplier returns.",
    ),

    # ── 19. Warning / inquiry messages about scam links (score 15) ──────────
    (
        "Suspicious Link in Forwarded/Inquiry Message",
        _any([
            r"(someone|they|he|she)\s+(sent|share[ds]?|forward[ed]*)\s+(me|this).{0,30}(link|url|site)",
            r"please\s+(check|verify|confirm)\s+(this|the)\s+(link|site|url)",
            r"(is\s+(this|it)\s+)?(real|legit|genuine|safe|scam|fraud)\??",
            r"(don.?t|do\s+not)\s+(click|open|visit).{0,30}(link|url|site)",
        ]),
        15,
        "The message references or forwards a suspicious link for verification — indicative of scam forwarding patterns.",
    ),

    # ── 20. Account profile update (vague) (score 15) ───────────────────────
    (
        "Vague Account Update Request",
        _any([
            r"account\s+(profile|detail[s]?|information)\s+(need[s]?\s+)?(update|verif|confirm)",
            r"update\s+your\s+(account|banking|profile)\s+(detail[s]?|information)",
            r"(visit|go\s+to).{0,30}(official\s+)?(bank|branch|app).{0,30}(assist|help|verify|update)",
            r"your\s+account\s+(may\s+be|will\s+be|is)\s+(at\s+risk|affected|impacted)",
        ]),
        15,
        "The message requests vague account profile updates — a common pretext used by phishing campaigns.",
    ),
]


SCORE_THRESHOLDS = {
    "Legitimate": (0, 29),
    "Suspicious":  (30, 54),
    "Scam":        (55, float("inf")),
}

RISK_LEVELS = {
    "Legitimate": "Low",
    "Suspicious":  "Medium",
    "Scam":        "High",
}


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def detect_scam(message_text: str) -> DetectionResult:
    """
    Analyse a message text using the rule-based scoring system.

    Returns a DetectionResult with:
      - predicted_label  : Legitimate | Suspicious | Scam
      - risk_score       : integer (cumulative weighted score, capped at 100)
      - risk_level       : Low | Medium | High
      - detected_indicators : list of triggered rule names
      - explanation      : human-readable reason string
    """
    if not message_text or not isinstance(message_text, str):
        return DetectionResult(
            predicted_label="Legitimate",
            risk_score=0,
            risk_level="Low",
            detected_indicators=[],
            explanation="No message content to analyse.",
        )

    total_score = 0
    triggered: List[Tuple[str, int, str]] = []   # (name, score, explanation)

    for name, pattern, score, explanation in RULES:
        if pattern.search(message_text):
            triggered.append((name, score, explanation))
            total_score += score

    # Cap score at 100 to keep scale meaningful
    total_score = min(total_score, 100)

    # Classify (Scam ≥ 55, Suspicious 30–54, Legitimate < 30)
    if total_score >= 55:
        label = "Scam"
    elif total_score >= 30:
        label = "Suspicious"
    else:
        label = "Legitimate"

    risk_level = RISK_LEVELS[label]
    detected_indicators = [t[0] for t in triggered]

    if triggered:
        rule_explanations = "; ".join(t[2] for t in triggered)
        explanation = (
            f"Classified as {label} (score {total_score}/100) because: {rule_explanations}"
        )
    else:
        explanation = (
            "No scam indicators detected. The message appears to be a normal, "
            "legitimate communication."
        )

    return DetectionResult(
        predicted_label=label,
        risk_score=total_score,
        risk_level=risk_level,
        detected_indicators=detected_indicators,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper returning a plain dict (for DataFrame use)
# ---------------------------------------------------------------------------

def detect_scam_dict(message_text: str) -> dict:
    r = detect_scam(message_text)
    return {
        "System_Prediction":    r.predicted_label,
        "System_Risk_Score":    r.risk_score,
        "System_Risk_Level":    r.risk_level,
        "Detected_Indicators":  "; ".join(r.detected_indicators) if r.detected_indicators else "None",
        "Explanation":          r.explanation,
    }
