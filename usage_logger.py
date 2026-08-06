"""
Lightweight, privacy-conscious usage logging.

Goal: let Rudra see basic usage (how many checks happen, roughly how many
distinct visitors, when) without storing anything that identifies a real
person.

Design choices, worth knowing (and worth citing in the written responses):

- The message text and sender field are NEVER logged here, no matter what.
  Once real people start pasting in real scam messages, that text can
  contain real names, addresses, or other personal details -- keeping it
  out of any log entirely is a hard rule, not a setting.

- The raw IP address is not stored either. IP addresses are treated as
  personal data under privacy laws like GDPR/CCPA, and logging them for a
  tool aimed at seniors sits uncomfortably close to the same "someone is
  tracking you" pattern this app exists to warn people about. Instead this
  stores a one-way HMAC-SHA256 hash of (secret salt + IP). The same
  visitor hashes to the same value across requests -- so distinct sessions
  and repeat visits can still be counted -- but the hash can't be turned
  back into the original IP without the salt.

- Honest caveat, not hidden: IPv4 has only ~4 billion possible addresses,
  so if the salt ever leaked, someone could brute-force a hash back to its
  IP in well under a second on ordinary hardware. The SECRET salt is what
  makes this meaningfully more private than logging raw IPs -- not the
  hash algorithm by itself. That's why the salt is generated once, written
  to a file kept OUT of git (`.session_salt`, listed in .gitignore), and
  never logged, printed, or committed anywhere.

- Deploying to a host with an ephemeral filesystem (Render's free tier,
  for example) means a locally-generated `.session_salt` file gets wiped
  on every redeploy/restart -- which would silently reset everyone's
  session hash each time. To avoid that, this module checks for a
  SESSION_SALT_HEX environment variable first, and only falls back to
  generating/reading a local file if that's not set. Set SESSION_SALT_HEX
  once in your hosting platform's environment variable settings (generate
  a value with `python3 -c "import secrets; print(secrets.token_hex(32))"`)
  and it'll survive redeploys. Local dev without that env var still works
  exactly as before, via the file.

- The usage LOG itself (logs/usage.log) still won't survive a redeploy on
  an ephemeral host, even with a stable salt -- only the salt is designed
  to persist. That's an accepted tradeoff for a free-tier deployment: the
  log is genuinely useful for a stretch of usage, just don't expect it to
  accumulate forever across every redeploy. A persistent disk (paid tier)
  or an external log store would fix this if it's ever needed.
"""
import datetime
import hashlib
import hmac
import json
import os
import pathlib

HERE = pathlib.Path(__file__).parent
SALT_FILE = HERE / ".session_salt"
LOG_DIR = HERE / "logs"
LOG_FILE = LOG_DIR / "usage.log"


def _load_or_create_salt():
    """A secret salt, stable across restarts (so the same visitor keeps
    hashing to the same value) and secret (so the hash can't be reversed
    without it).

    Prefers the SESSION_SALT_HEX environment variable -- set this on any
    host with an ephemeral filesystem (see module docstring) so the salt
    survives redeploys. Falls back to a local file, generated once on
    first run, for local development.
    """
    env_salt = os.environ.get("SESSION_SALT_HEX")
    if env_salt:
        return bytes.fromhex(env_salt)
    if SALT_FILE.exists():
        return SALT_FILE.read_bytes()
    salt = os.urandom(32)
    SALT_FILE.write_bytes(salt)
    try:
        os.chmod(SALT_FILE, 0o600)  # owner read/write only
    except OSError:
        pass  # not fatal (e.g. some hosting filesystems don't support chmod)
    return salt


_SALT = _load_or_create_salt()


def hash_ip(ip):
    """One-way, salted hash of an IP address. Returns None if no IP was
    available (e.g. running behind a misconfigured proxy)."""
    if not ip:
        return None
    digest = hmac.new(_SALT, ip.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:16]  # 64 bits is plenty to tell sessions apart; keeps log lines short


def log_usage(route, session_hash, risk_level=None, outcome="ok"):
    """Append one line to logs/usage.log.

    Only ever pass route/session_hash/risk_level/outcome in here -- never
    message text or sender, which is the whole point of this module.
    """
    LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "session": session_hash,
        "route": route,
        "risk_level": risk_level,
        "outcome": outcome,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
