import sys, pathlib, random, re
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cases import SCAMS, SAFE
from model.scam_detector import ScamDetector
d = ScamDetector(); rng = random.Random(3)
def leet(t): return t.translate(str.maketrans("aeio", "4310")) if rng.random() < 2 else t
def typos(t):
    w = t.split(); 
    for i in rng.sample(range(len(w)), max(1, len(w)//6)):
        x = w[i]
        if len(x) > 4: j = rng.randrange(1, len(x)-2); w[i] = x[:j] + x[j+1] + x[j] + x[j+2:]
    return " ".join(w)
MUT = {
 "upper": str.upper, "lower": str.lower, "typos": typos,
 "emoji": lambda t: "🚨 " + t + " 🙏🙏",
 "filler": lambda t: "Hope you are doing well! " + t + " Thanks so much, take care.",
 "extra_spaces": lambda t: re.sub(r" ", "  ", t),
 "newlines": lambda t: t.replace(". ", ".\n\n"),
 "no_punct": lambda t: re.sub(r"[.,!?]", "", t),
}
def rate(texts, f, scam=True):
    n = 0
    for t in texts:
        lvl = d.analyze(f(t))["risk_level"]
        n += (lvl != "low") if scam else (lvl == "low")
    return n
print(f"{'mutation':14} scams flagged   safe still safe")
for name, f in MUT.items():
    print(f"{name:14} {rate(SCAMS,f)}/{len(SCAMS)}          {rate(SAFE,f,False)}/{len(SAFE)}")
