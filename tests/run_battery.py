import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cases import SCAMS, SAFE, SENDER_CASES
from model.scam_detector import ScamDetector
d = ScamDetector()
miss = fp = 0
print("--- scams that were called LOW (missed) or MEDIUM (soft) ---")
for t in SCAMS:
    r = d.analyze(t)
    if r["risk_level"] == "low": miss += 1; print("MISSED", r["probability"], t[:80])
    elif r["risk_level"] == "medium": print("soft  ", r["probability"], t[:80])
print("--- safe messages flagged ---")
for t in SAFE:
    r = d.analyze(t)
    if r["risk_level"] != "low": fp += 1; print("FALSE ALARM", r["risk_level"], r["probability"], t[:80])
print(f"scams: {len(SCAMS)-miss}/{len(SCAMS)} flagged | safe: {len(SAFE)-fp}/{len(SAFE)} passed")
order = {"low": 0, "medium": 1, "high": 2}
for t, s, want in SENDER_CASES:
    r = d.analyze(t, sender=s); ok = order[r["risk_level"]] >= order[want] if want != "low" else r["risk_level"] == "low"
    print("OK  " if ok else "FAIL", s, "->", r["risk_level"])
