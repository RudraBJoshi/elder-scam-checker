"""
ScamDetector: loads the trained TF-IDF + Logistic Regression model and turns a
raw prediction into the plain-language explanation the senior-friendly UI shows.

Two layers of "why":
  1. Model-driven: the TF-IDF n-grams from the actual message that contributed
     most to the "scam" prediction (via TfidfVectorizer feature weight * the
     Logistic Regression coefficient for that feature -- a standard linear-model
     explainability trick, no extra library needed).
  2. Rule-driven: a small curated dictionary of well-known scam tactics
     (urgency, secrecy, unusual payment methods, authority impersonation,
     too-good-to-be-true, requests for personal info) and the 8 IC3-grounded
     scam categories from the dataset (generate_elder_examples.py), used to
     surface a plain-language "this looks like a X" category tag with
     tailored next-step advice. This keeps explanations readable for seniors
     instead of showing raw n-grams as the only "why".
"""
import json
import pathlib
import re

import joblib

try:
    from model.feature_utils import build_features
except ImportError:
    from feature_utils import build_features

HERE = pathlib.Path(__file__).parent

# --- Tactic red flags (shown as "Why we flagged this") -------------------------
TACTIC_FLAGS = [
    {
        "key": "urgency",
        "label": "Creates urgency or a countdown",
        "keywords": ["immediately", "urgent", "act now", "within 24 hours",
                     "expires today", "before it's too late", "final notice", "hurry",
                     "act immediately", "respond immediately"],
    },
    {
        "key": "secrecy",
        "label": "Asks you to keep it a secret",
        "keywords": ["don't tell", "do not tell", "keep this between us", "confidential",
                     "don't call my parents", "please don't tell", "our secret",
                     "keep it private", "keep this private", "keep it quiet", "don't tell anyone",
                     "do not tell anyone", "don't mention this", "do not mention this", "between us"],
    },
    {
        "key": "unusual_payment",
        "label": "Wants payment by gift card, wire transfer, or crypto",
        "keywords": ["gift card", "gift cards", "wire transfer", "moneygram", "western union",
                     "bitcoin", "cryptocurrency", "crypto", "usdt", "redemption code",
                     "amazon cards", "amazon card", "itunes cards", "itunes card", "google play cards",
                     "google play card", "apple cards", "steam cards", "target cards", "prepaid card",
                     "prepaid cards", "read me the codes", "send me the codes", "send the codes"],
    },
    {
        "key": "authority",
        "label": "Claims to be a government agency or well-known company",
        "keywords": ["irs", "social security", "medicare", "u.s. marshals", "department of treasury",
                     "microsoft", "apple", "amazon", "norton", "geek squad", "paypal", "police",
                     "google", "facebook", "instagram", "whatsapp", "netflix", "your bank",
                     "officer", "warrant", "arrest"],
    },
    {
        "key": "threat",
        "label": "Threatens arrest, a lawsuit, or losing your account",
        "keywords": ["lawsuit", "arrest", "warrant", "suspended", "terminated", "legal action",
                     "account will be closed", "jail"],
    },
    {
        "key": "too_good",
        "label": "Promises a prize or unrealistic returns",
        "keywords": ["congratulations", "you've won", "you have won", "guaranteed return",
                     "guaranteed returns", "zero risk", "winner", "sweepstakes", "jackpot", "free money"],
    },
    {
        "key": "personal_info",
        "label": "Asks you to verify or confirm personal information",
        "keywords": ["verify your", "confirm your", "social security number", "account number",
                     "date of birth", "click the link", "click here", "log in to verify"],
    },
    {
        "key": "account_pretext",
        "label": "Says there is a problem with your account or device that only they can fix",
        "keywords": [],
        "regex": r"\b(?:fix|unlock|restore|repair|reactivate)\b[^.!?\n]{0,25}"
                 r"\b(?:your|ur)\s+(?:account|acount|acocunt|accont|password|computer|device|phone)\b",
    },
    {
        "key": "emergency_money",
        "label": "Claims someone is in danger and demands money right away",
        "keywords": ["kidnap", "kidnapped", "kidnapping", "kidnapper", "kidnappers", "ransom",
                     "hostage", "they have me", "they will hurt", "i'm in danger",
                     "i am in danger", "hurt me"],
    },
    {
        "key": "money_request",
        "label": "Asks you to send money",
        "keywords": [],
        "regex": r"\b(?:need|send|wire|pay|give|transfer|owe|have|get|lend|borrow|loan)\b[^.!?\n]{0,40}"
                 r"(?:\$\s?\d|\b\d[\d,.]*\s?(?:k|grand|dollars|usd|bucks)\b)",
    },
    {
        "key": "remote_access",
        "label": "Asks for remote access to your computer",
        "keywords": ["remote access", "allow us access", "download teamviewer", "anydesk",
                     "restart your computer", "do not restart"],
    },
]

# --- Scam categories (matches dataset's scam_type field) -----------------------
SCAM_TYPES = {
    "virtual_kidnapping": {
        "label": "Emergency / Kidnapping Money Demand",
        "keywords": ["kidnap", "kidnapped", "kidnapping", "kidnapper", "kidnappers", "ransom",
                     "hostage", "they have me", "help me"],
        "advice": "Do not send money. \"Virtual kidnapping\" scams use a frightening message to rush "
                  "you into paying before you can think. Contact the person directly on a number "
                  "you already have, and if you believe someone is truly in danger, call 911 right "
                  "away -- real kidnappers do not ask for money by text.",
    },
    "grandparent_scam": {
        "label": "Grandparent Scam",
        "keywords": ["grandma", "grandpa", "it's me", "bail", "accident", "jail",
                     "don't tell mom", "don't tell dad", "hospital", "broke my phone"],
        "advice": "Hang up or stop replying, then call your family member directly using "
                  "a phone number you already have saved for them -- not any number given "
                  "in the message. Real emergencies rarely require secrecy or gift cards.",
    },
    "government_impersonation": {
        "label": "Government Impersonation",
        "keywords": ["irs", "social security", "medicare", "u.s. marshals", "department of treasury",
                     "warrant", "jury duty", "officer"],
        "advice": "Real government agencies do not call, text, or email demanding immediate "
                  "payment or threatening arrest. Hang up and, if unsure, call the agency "
                  "directly using the number on their official website (not one from this message).",
    },
    "tech_support_scam": {
        "label": "Tech Support Scam",
        "keywords": ["virus", "infected", "technical support", "remote access", "antivirus",
                     "malware", "microsoft support", "apple support", "geek squad"],
        "advice": "Legitimate tech companies don't send pop-ups or unsolicited calls asking to "
                  "remote into your computer. Close the window or hang up, and don't install "
                  "anything or give anyone remote access.",
    },
    "romance_scam": {
        "label": "Romance Scam",
        "keywords": ["my darling", "sweetheart", "my love", "never felt this close",
                     "trust you", "meet you", "customs fee", "oil rig"],
        "advice": "Be cautious with anyone met online who asks for money, especially before "
                  "meeting in person. Talk to a trusted family member before sending anything, "
                  "and never send money to someone you haven't met face to face.",
    },
    "investment_fraud": {
        "label": "Investment Fraud",
        "keywords": ["guaranteed return", "guaranteed returns", "trading group", "zero risk",
                     "pre-ipo", "trading bot", "portfolio", "crypto opportunity"],
        "advice": "No legitimate investment guarantees high returns with no risk. Don't send "
                  "money or crypto based on a message or cold call -- check with a licensed "
                  "financial advisor or a trusted family member first.",
    },
    "business_email_compromise": {
        "label": "Gift Card / Favor Scam",
        "keywords": ["pick up", "gift card", "redemption codes", "stuck in a meeting",
                     "can't talk right now", "keep this between us"],
        "advice": "This impersonates someone you know (a boss, coworker, or relative) asking "
                  "for a quick favor buying gift cards. Contact that person directly, by phone "
                  "or in person, before buying or sending anything.",
    },
    "lottery_sweepstakes": {
        "label": "Lottery / Sweepstakes Scam",
        "keywords": ["congratulations", "you've won", "you have won", "sweepstakes", "jackpot",
                     "claim your prize", "processing fee"],
        "advice": "You can't win a lottery or sweepstakes you never entered, and real prizes "
                  "never require you to pay a fee first. Don't send any money or gift cards to "
                  "'release' a prize.",
    },
    "phishing_spoofing": {
        "label": "Phishing / Fake Company Alert",
        "keywords": ["account has been locked", "verify your identity", "update your billing",
                     "unusual sign-in", "click the link", "account has been temporarily frozen",
                     "account has been suspended"],
        "advice": "Don't click links in unexpected messages claiming to be your bank or another "
                  "company. Instead, open your bank's app directly or type the company's website "
                  "address yourself to check your account.",
    },
    "package_delivery_scam": {
        "label": "Fake Package Delivery Notice",
        "keywords": ["package could not be delivered", "customs fee", "reschedule the delivery",
                     "reschedule delivery", "unpaid fee", "manage delivery", "in clearance processing",
                     "parcel is pending", "confirm your shipment"],
        "advice": "Real delivery companies (USPS, FedEx, UPS, and others) don't text asking you "
                  "to pay a small fee through a link to release a package. Don't click the link "
                  "-- track your package directly through the carrier's official app or website "
                  "instead.",
    },
    "job_scam": {
        "label": "Fake Job or Business Opportunity",
        # Added 2026-08-05 after a real unsolicited "sourcing agent for a
        # pharmaceutical company" recruitment email was tested through the
        # live app: the model already scored it 0.999 (correctly suspicious)
        # but no category matched, so it capped at "medium" instead of
        # "high" -- see app_build_notes.md.
        "keywords": ["sourcing agent", "sourcing efforts", "trustworthy individual",
                     "financial incentives", "formal agreement", "corporate offer",
                     "full corporate offer", "name and address", "whatsapp",
                     "hiring manager", "work from home opportunity", "no experience necessary",
                     "shipping agent", "reshipping", "package handling",
                     # Added 2026-08-06 after a second real test message: a
                     # "YouTube Channel Growth Partner" task-scam text with
                     # the same underlying pattern (unsolicited, vague task,
                     # unrealistic pay, contact via bare phone number) but
                     # different specific wording -- same category, broader
                     # keyword coverage.
                     "channel growth", "click-through rate", "click-through rates",
                     "daily earnings", "paid trial", "instant daily incentives",
                     "no prior background", "spots available"],
        "advice": "Real employers don't recruit out of the blue by email and ask for your home "
                  "address and phone number before describing the actual job. Don't reply with "
                  "personal information -- independently look up the company (a real business "
                  "should have reviews and a verifiable history), and never accept a job that "
                  "involves receiving, forwarding, or reshipping packages or money for someone else.",
    },
}


GENERIC_ADVICE = ("Do not send money, gift cards, or personal information, and do not click any "
                  "links. Contact the person or company directly using a phone number you already "
                  "have -- not one from this message.")

NEXT_STEPS = [
    "Do not reply, click links, or send money or gift cards.",
    "Call the person or company using a number you already have.",
    "Tell a family member or friend you trust before you do anything.",
]


def _find_keyword_hits(text_lower, keywords):
    """Word-boundary-aware substring match. Plain `kw in text_lower` was
    matching short keywords inside unrelated words -- e.g. "irs" (for the
    IRS) matched inside "first", falsely flagging any message that used the
    word "first" as government impersonation. \\b anchors keywords to whole
    words/phrases so "irs" only matches "irs" as its own token."""
    hits = []
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            hits.append(kw)
    return hits


# --- Sender checks (optional -- only used if the user provides an email/phone) -
# NOT exhaustive. This is a small starter list of domains commonly impersonated
# in elder-scam messages (per the dataset categories above). Add to this list
# for organizations relevant to your own users (their bank, pharmacy, doctor's
# office, etc.) -- an easy, high-value place to extend this app.
KNOWN_OFFICIAL_DOMAINS = {
    "irs.gov": "the IRS",
    "ssa.gov": "the Social Security Administration",
    "medicare.gov": "Medicare",
    "usps.com": "USPS",
    "fedex.com": "FedEx",
    "ups.com": "UPS",
    "amazon.com": "Amazon",
    "paypal.com": "PayPal",
    "microsoft.com": "Microsoft",
    "apple.com": "Apple",
    "chase.com": "Chase",
    "bankofamerica.com": "Bank of America",
    "wellsfargo.com": "Wells Fargo",
    "citibank.com": "Citibank",
    "netflix.com": "Netflix",
    "walmart.com": "Walmart",
    "target.com": "Target",
    "costco.com": "Costco",
    "capitalone.com": "Capital One",
    "usbank.com": "U.S. Bank",
    "verizon.com": "Verizon",
    "att.com": "AT&T",
    "comcast.com": "Comcast",
    "cvs.com": "CVS",
    "walgreens.com": "Walgreens",
    "kaiserpermanente.org": "Kaiser Permanente",
    "sharp.com": "Sharp HealthCare",
    "norton.com": "Norton",
    "mcafee.com": "McAfee",
    "ebay.com": "eBay",
    "dhl.com": "DHL",
    "usa.gov": "USA.gov",
    "va.gov": "the VA",
}

# Words that mean "this domain is trying to look like <org>" when they appear in
# a domain that is NOT that organization's official one.
BRAND_WORDS = {
    "irs": "the IRS", "ssa": "Social Security", "socialsecurity": "Social Security",
    "medicare": "Medicare", "usps": "USPS", "fedex": "FedEx", "ups": "UPS", "dhl": "DHL",
    "amazon": "Amazon", "paypal": "PayPal", "microsoft": "Microsoft", "apple": "Apple",
    "chase": "Chase", "wellsfargo": "Wells Fargo", "bankofamerica": "Bank of America",
    "citibank": "Citibank", "capitalone": "Capital One", "netflix": "Netflix",
    "walmart": "Walmart", "norton": "Norton", "mcafee": "McAfee", "geeksquad": "Geek Squad",
    "usbank": "U.S. Bank", "verizon": "Verizon", "comcast": "Comcast",
}

URL_SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "rb.gy", "is.gd", "cutt.ly", "ow.ly",
                  "shorturl.at", "buff.ly", "tiny.cc", "goo.gl", "rebrand.ly", "bl.ink"}

# Cheap, throwaway TLDs that show up in a large share of reported smishing links.
SUSPICIOUS_TLDS = {"top", "xyz", "click", "online", "help", "site", "live", "icu", "buzz",
                   "tk", "cc", "support", "link", "cfd", "sbs", "info", "rest", "monster",
                   "vip", "shop", "store", "work", "today", "cam"}

_TLDS = ("com|net|org|gov|edu|us|io|co|me|ly|gl|info|biz|top|xyz|click|online|help|site|live|"
         "icu|buzz|tk|cc|support|link|cfd|sbs|rest|monster|vip|shop|store|work|today|cam|app|"
         "dev|page|cloud|email|security|services|solutions|center")
URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+|"
    r"\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:" + _TLDS + r")\b(?:/[^\s<>\"']*)?",
    re.I,
)
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "@": "a", "$": "s"})


def _host_of(token):
    token = re.sub(r"^https?://", "", token.strip().rstrip(".,;:!?)\"'"), flags=re.I)
    host = re.split(r"[/?#]", token, maxsplit=1)[0]
    host = host.split("@")[-1].split(":")[0].lower()
    return host[4:] if host.startswith("www.") else host


def _is_official(host):
    return any(host == d or host.endswith("." + d) for d in KNOWN_OFFICIAL_DOMAINS)


def _domain_signals(host):
    """Reasons a single web/email domain looks deceptive. Returns a list of
    plain-language strings; empty means nothing concrete was found."""
    if not host or _is_official(host):
        return []
    reasons = []
    labels = host.split(".")
    tld = labels[-1]
    registered = ".".join(labels[-2:])

    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        return [f"The link points to a raw number address ({host}) instead of a real website name."]
    if registered in URL_SHORTENERS or host in URL_SHORTENERS:
        return [f"The link ({host}) is a shortened link, which hides where it really goes."]

    squashed = re.sub(r"[^a-z0-9]", "", host.translate(LEET))
    for word, org in BRAND_WORDS.items():
        if word in squashed:
            reasons.append(f"The address \"{host}\" is made to look like {org}, but it is not {org}'s real website.")
            break
    if tld in SUSPICIOUS_TLDS and not reasons:
        reasons.append(f"The address \"{host}\" ends in \".{tld}\", a kind of address scammers often use.")
    return reasons


def _link_signals(text):
    seen, signals = set(), []
    for m in URL_RE.finditer(text):
        host = _host_of(m.group(0))
        if host in seen or "." not in host:
            continue
        seen.add(host)
        for reason in _domain_signals(host):
            signals.append({"type": "suspicious", "label": reason})
    return signals

FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "icloud.com", "live.com", "mail.com", "protonmail.com", "yandex.com",
}

GOV_AGENCY_MENTIONS = ["irs", "social security", "medicare", "department of treasury", "u.s. marshals"]

EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
PHONE_RE = re.compile(r"[\d][\d\-\s().+]{6,}\d")


def _sender_signals(sender, text_lower):
    """Returns a list of {"type": "suspicious"|"reassuring"|"info", "label": str}
    based on an optional email address or phone number the user provides for
    who sent the message. This is intentionally conservative: it only speaks up
    when it recognizes something concrete, never guesses."""
    signals = []
    if not sender:
        return signals
    sender = sender.strip()

    email_match = EMAIL_RE.search(sender)
    if email_match:
        domain = email_match.group(1).lower()

        known_match = None
        for official_domain, org_name in KNOWN_OFFICIAL_DOMAINS.items():
            if domain == official_domain or domain.endswith("." + official_domain):
                known_match = org_name
                break

        if known_match:
            signals.append({
                "type": "reassuring",
                "label": f"The sender's email domain matches {known_match}'s official domain."
            })
        else:
            for reason in _domain_signals(domain):
                signals.append({"type": "suspicious", "label": "Sender email: " + reason})
            if domain in FREE_EMAIL_PROVIDERS and any(k in text_lower for k in GOV_AGENCY_MENTIONS):
                signals.append({
                    "type": "suspicious",
                    "label": f"Claims to be from a government agency, but was sent from a free "
                              f"email address ({domain}) instead of an official agency domain."
                })
            if any(k in text_lower for k in GOV_AGENCY_MENTIONS) and not domain.endswith(".gov"):
                signals.append({
                    "type": "suspicious",
                    "label": "Claims to be a U.S. government agency, but real government agencies "
                              "use \".gov\" email addresses -- this one doesn't."
                })
        return signals

    phone_match = PHONE_RE.search(sender)
    if phone_match:
        signals.append({
            "type": "info",
            "label": "We can't check phone numbers against a directory. If you don't recognize "
                     "this number, look up the organization's number yourself (from a bill, card, "
                     "or official website) instead of calling back the number in the message."
        })

    return signals


class ScamDetector:
    def __init__(self, model_dir=None):
        model_dir = pathlib.Path(model_dir) if model_dir else HERE
        self.word_vectorizer = joblib.load(model_dir / "word_vectorizer.pkl")
        self.char_vectorizer = joblib.load(model_dir / "char_vectorizer.pkl")
        self.classifier = joblib.load(model_dir / "classifier.pkl")
        metrics_path = model_dir / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    def _predict_proba(self, text):
        x = build_features(self.word_vectorizer, self.char_vectorizer, [text])
        return float(self.classifier.predict_proba(x)[0][1])

    def _top_contributing_phrases(self, text, top_n=5):
        """Return the actual WORD phrases from THIS message that pushed the
        prediction toward 'scam', ranked by TF-IDF weight * model coefficient.
        Only looks at the word-vectorizer's slice of the model -- the char
        n-gram and structural features also feed the prediction but don't
        correspond to readable "phrases", so they're left out of this
        particular explanation (it's kept as API-level data, not shown
        directly in the UI, which relies on the tactic/category layer)."""
        x = self.word_vectorizer.transform([text])
        feature_names = self.word_vectorizer.get_feature_names_out()
        coefs = self.classifier.coef_[0][: len(feature_names)]  # word features come first

        contributions = []
        # x is a sparse row vector; iterate only its nonzero entries
        for idx in x.nonzero()[1]:
            weight = x[0, idx] * coefs[idx]
            if weight > 0:
                contributions.append((feature_names[idx], weight))

        contributions.sort(key=lambda pair: pair[1], reverse=True)
        phrases = [ph for ph, _ in contributions
                   if re.search(r"[a-z]{4,}", ph) and not re.fullmatch(r"[\d\s,.$]+", ph)]
        return phrases[:top_n]

    def _matched_tactics(self, text_lower):
        matched = []
        for tactic in TACTIC_FLAGS:
            hits = _find_keyword_hits(text_lower, tactic["keywords"])
            if tactic.get("regex") and re.search(tactic["regex"], text_lower):
                hits = hits or ["pattern"]
            if hits:
                matched.append({"key": tactic["key"], "label": tactic["label"], "matched": hits[:3]})
        return matched

    def _likely_scam_type(self, text_lower):
        best_key, best_score = None, 0
        for key, info in SCAM_TYPES.items():
            hits = _find_keyword_hits(text_lower, info["keywords"])
            if len(hits) > best_score:
                best_key, best_score = key, len(hits)
        if best_key and best_score >= 1:
            info = SCAM_TYPES[best_key]
            return {"key": best_key, "label": info["label"], "advice": info["advice"],
                    "hits": best_score}
        return None

    # A bare brand/agency name ("Amazon", "Medicare") or a single loose category
    # keyword appears in plenty of legitimate notices, so on its own it is not
    # a red flag; it only counts alongside a stronger signal.
    WEAK_TACTICS = {"authority", "money_request"}
    # With no concrete red flag at all, a mid-range model score is too shaky to
    # show the user as a warning -- below this we call it safe.
    NO_EVIDENCE_SAFE_BELOW = 0.35

    @classmethod
    def _has_evidence(cls, tactics, likely_type, suspicious_signals):
        if suspicious_signals:
            return True
        if any(t["key"] not in cls.WEAK_TACTICS for t in tactics):
            return True
        if len(tactics) >= 2:
            return True
        return bool(likely_type and likely_type["hits"] >= 2)

    @classmethod
    def _risk_tier(cls, prob, has_evidence, sender_reassured, tactic_count=0):
        """Turn the raw model probability into a plain-language tier.

        Important design choice: a "high risk / SCAM" verdict requires at least
        one concrete, explainable red flag (a matched tactic, a matched scam
        category, or a suspicious sender signal) -- not just a raw model score.
        A model that's confident but can't point to any actual red flag gets
        demoted to a more honest, cautious tier instead of a false alarm.
        A verified-legitimate sender domain can pull an otherwise-risky score
        down further, since a confirmed official domain is strong evidence.
        """
        if sender_reassured and not has_evidence:
            # The wording alone raised some suspicion, but the sender's domain
            # checks out against a known official domain and nothing else
            # (no gift cards, threats, secrecy, etc.) actually looks wrong.
            return "low", "This looks safe"

        # Several independent warning signs is strong evidence even if the model
        # score is low, so it must not hinge on a thin margin around a cutoff.
        if tactic_count >= 3 or (tactic_count >= 2 and prob >= 0.20):
            return "high", "This looks like a SCAM"

        if prob >= 0.70:
            if has_evidence:
                return "high", "This looks like a SCAM"
            return "medium", "This looks suspicious — check before you act"
        elif prob >= 0.35:
            if tactic_count >= 2:
                return "high", "This looks like a SCAM"
            if has_evidence:
                return "medium", "Be careful — this has warning signs"
            if prob >= cls.NO_EVIDENCE_SAFE_BELOW:
                return "medium", "This looks suspicious — check before you act"
            return "low", "This looks safe"
        else:
            return "low", "This looks safe"

    def analyze(self, text, sender=None):
        text = (text or "").strip()
        if not text:
            return None

        prob = self._predict_proba(text)

        text_lower = re.sub(r"\bur\b", "your", re.sub(r"\bu\b", "you", text.lower()))
        tactics = self._matched_tactics(text_lower)
        likely_type = self._likely_scam_type(text_lower)
        sender_signals = _sender_signals(sender, text_lower) + _link_signals(text)

        suspicious_sender = [s for s in sender_signals if s["type"] == "suspicious"]
        reassuring_sender = [s for s in sender_signals if s["type"] == "reassuring"]

        has_evidence = self._has_evidence(tactics, likely_type, suspicious_sender)
        sender_reassured = bool(reassuring_sender)

        risk_level, risk_label = self._risk_tier(prob, has_evidence, sender_reassured, len(tactics))

        # A lookalike/shortened link or spoofed sender domain is concrete evidence on its
        # own, even when the message wording is bland and the model score is low.
        if suspicious_sender and risk_level == "low":
            risk_level, risk_label = "medium", "Be careful — this has warning signs"

        # Don't show flagged tactics/categories on a verdict we're calling safe --
        # a stray keyword match shouldn't contradict a "this looks safe" headline.
        if risk_level == "low":
            tactics = []
            likely_type = None

        top_phrases = self._top_contributing_phrases(text, top_n=5) if risk_level != "low" else []

        reasons = [t["label"] for t in tactics]
        reasons += [s["label"] for s in sender_signals if s["type"] == "suspicious"]
        if risk_level != "low" and not reasons:
            quoted = ", ".join(f"\u201c{ph}\u201d" for ph in top_phrases[:3])
            reasons.append("The wording is a lot like scam messages we have seen before"
                           + (f" (for example: {quoted})." if quoted else "."))

        if risk_level == "low":
            advice = None
            next_steps = []
        else:
            advice = likely_type["advice"] if likely_type else GENERIC_ADVICE
            next_steps = list(NEXT_STEPS)

        return {
            "reasons": reasons,                 # plain-language "why", always non-empty unless low risk
            "advice": advice,                   # what to do (category-specific when known)
            "next_steps": next_steps,           # short universal checklist for medium/high
            "reassurance": [s["label"] for s in reassuring_sender] if risk_level == "low" else [],
            "risk_level": risk_level,          # "high" | "medium" | "low"
            "risk_label": risk_label,          # plain-language headline
            "probability": round(prob, 3),      # 0-1, model's raw confidence this is a scam
            "top_phrases": top_phrases,         # actual phrases from the message that drove the score
            "tactics": tactics,                 # matched red-flag tactic categories
            "likely_type": likely_type,         # best-guess scam category + tailored advice, or None
            "sender_signals": sender_signals,   # suspicious/reassuring/info notes about the sender
        }
