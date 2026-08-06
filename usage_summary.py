"""
Print a quick summary of logs/usage.log -- how many messages were checked,
how many distinct (hashed) sessions used the app, and the risk-level
breakdown. Run anytime with:

    python3 usage_summary.py

Reads the same privacy-conscious log usage_logger.py writes: hashed
session IDs only, no IP addresses, no message text.
"""
import collections
import json
import pathlib

LOG_FILE = pathlib.Path(__file__).parent / "logs" / "usage.log"


def main():
    if not LOG_FILE.exists():
        print("No usage logged yet (logs/usage.log doesn't exist).")
        return

    entries = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]
    if not entries:
        print("logs/usage.log is empty.")
        return

    sessions = {e["session"] for e in entries if e.get("session")}
    checks = [e for e in entries if e["route"] == "/check"]
    completed_checks = [e for e in checks if e.get("outcome") == "ok"]
    by_risk = collections.Counter(e.get("risk_level") for e in completed_checks)
    by_day = collections.Counter(e["ts"][:10] for e in entries)

    print(f"Total requests logged: {len(entries)}")
    print(f"Distinct sessions (hashed, not identifiable): {len(sessions)}")
    print(f"Messages checked: {len(checks)} ({len(completed_checks)} completed)")
    print()
    print("Risk level breakdown:")
    for level, count in by_risk.most_common():
        print(f"  {level:8s} {count}")
    print()
    print("Requests by day:")
    for day, count in sorted(by_day.items()):
        print(f"  {day}  {count}")


if __name__ == "__main__":
    main()
