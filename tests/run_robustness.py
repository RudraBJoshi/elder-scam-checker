import sys, time, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from app import app
app.config["TESTING"] = True
c = app.test_client()
fails = []
def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond: fails.append(name)

def post_json(payload, **kw): return c.post("/check", json=payload, **kw)

# routes
check("GET /", c.get("/").status_code == 200)
check("GET /about", c.get("/about").status_code == 200)
check("GET /check -> 405", c.get("/check").status_code == 405)
check("404 page", c.get("/nope").status_code == 404)
# empty / whitespace
for name, payload in [("empty", {"message": ""}), ("whitespace", {"message": "   \n\t "}), ("missing", {}), ("null", {"message": None})]:
    r = post_json(payload); check(f"json {name} -> 400", r.status_code == 400, r.status_code)
r = c.post("/check", data={"message": ""}); check("form empty renders error", r.status_code == 200 and "Please paste" in r.get_data(as_text=True))
# bad JSON body / wrong types
r = c.post("/check", data="not json", content_type="application/json"); check("bad json no 500", r.status_code < 500, r.status_code)
for name, payload in [("int message", {"message": 12345}), ("list message", {"message": ["a"]}), ("dict message", {"message": {"a": 1}}),
                       ("int sender", {"message": "hello", "sender": 5}), ("list sender", {"message": "hello", "sender": ["x"]})]:
    r = post_json(payload); check(f"{name} no 500", r.status_code < 500, r.status_code)
# unicode / odd text
for name, t in [("emoji", "🎉🎉 You won!!! 💰💰 send $500 gift cards 🎁"), ("rtl", "مرحبا أنا جدك أحتاج 500 دولار"), ("cjk", "你好我是你的孙子，请汇款"),
                ("null byte", "hello\x00world $500 gift card"), ("only punctuation", "!!!???..."), ("only digits", "1234567890"), ("one char", "a"),
                ("mixed newlines", "line1\r\nline2\rline3\nsend $500 by gift card")]:
    r = post_json({"message": t}); check(f"unicode/odd: {name}", r.status_code == 200 and "risk_level" in r.get_json(), r.status_code)
# length
r = post_json({"message": "hello friend " * 1000}); check("very long message ok", r.status_code == 200)
r = post_json({"message": "x" * 100000}); check("100k chars ok", r.status_code == 200)
# regex slowness (ReDoS)
worst = {"a"*5000: "a repeated", "call " + "1-800 " * 500: "call spam", "$" * 3000: "dollars", "refund " * 400 + "call " + "9" * 2000: "callback",
         "need " + "a " * 1500: "need spam", "www." + "a." * 1500 + "com": "url dots", "http://" + "a" * 3000: "long host",
         "earn " + "1," * 1500: "earn", "fix " + "a" * 2900 + " your account": "fix", ("a-" * 1400) + ".com/": "hyphen host"}
for t, name in worst.items():
    t0 = time.time(); r = post_json({"message": t[:3000]}); dt = time.time() - t0
    check(f"ReDoS {name} ({dt:.2f}s)", r.status_code == 200 and dt < 2.0, dt)
# sender field abuse
for name, s in [("very long sender", "a" * 5000 + "@x.com"), ("sql", "'; DROP TABLE users;--@evil.com"), ("weird", "@@@@"), ("only @", "@"),
                ("unicode email", "аlerts@chаse.com"), ("phone", "+1 (858) 555-0199"), ("trailing dot domain", "a@chase.com."), ("uppercase", "ALERTS@CHASE.COM")]:
    r = post_json({"message": "Your statement is ready.", "sender": s}); check(f"sender: {name}", r.status_code == 200, r.status_code)
# XSS: reflected content must be escaped in both render paths
xss = '<script>alert(1)</script><img src=x onerror=alert(2)>'
r = c.post("/check", data={"message": xss + " send $500 gift card", "sender": xss})
html = r.get_data(as_text=True)
check("XSS escaped in form path", "<script>alert(1)</script>" not in html and "onerror=alert(2)>" not in html.replace("&lt;img src=x onerror=alert(2)&gt;", ""))
j = post_json({"message": xss + " send $500 gift card"}).get_json()
blob = json.dumps(j)
check("JSON never echoes raw HTML unescaped-by-client needs (labels only)", "<script>" not in json.dumps(j["reasons"]))
# sender lookalike edge: legit subdomain and tricky suffix
for s, want in [("no-reply@accounts.google.com", "not-suspicious"), ("x@chase.com.evil.top", "suspicious"), ("x@notchase.com", "any")]:
    r = post_json({"message": "Your statement is ready.", "sender": s}).get_json()
    susp = any(x["type"] == "suspicious" for x in r["sender_signals"])
    ok = (want == "suspicious" and susp) or (want == "not-suspicious" and not susp) or want == "any"
    check(f"sender domain {s} -> {'suspicious' if susp else 'clean'}", ok)
# consistency: same input, same output
a = post_json({"message": "Send $500 in gift cards now"}).get_json(); b = post_json({"message": "Send $500 in gift cards now"}).get_json()
check("deterministic", a == b)
# case / whitespace invariance
x = post_json({"message": "SEND $500 IN GIFT CARDS NOW"}).get_json()["risk_level"]; y = post_json({"message": "send $500 in gift cards now"}).get_json()["risk_level"]
check("case invariant risk level", x == y, (x, y))
print("\nFAILURES:", fails if fails else "none")
