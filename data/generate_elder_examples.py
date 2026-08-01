"""
Hand-authored synthetic examples for the elder-scam-detector dataset.

v2 (expanded, 2026-07-29): grew out of a real-world false positive Rudra found
(a legitimate healthcare-appointment notification was called a SCAM) and an
error-analysis pass that showed why: the dataset's "legitimate" class was 100%
casual SMS-style text, with zero formal/institutional notifications, and the
8 IC3-grounded scam categories only had 3-4 templates each (~178 rows total),
which also meant train/test splits could see near-duplicate rows from the same
template on both sides (see rebuild_dataset.py's grouped split, which needs
the template_id this file now tracks).

This version:
  - Adds a NEW "legitimate_formal" category: original examples of ordinary
    automated notifications (health, banking, delivery, subscriptions,
    utilities, civic/library) that use formal language but have none of the
    actual scam hallmarks -- directly targets the false-positive root cause.
  - Roughly doubles the number of distinct templates per scam category (not
    just more variable combinations of the same few templates), so the model
    sees more genuine phrasing variety and no single template dominates a
    category.
  - Tags every row with a template_id (the literal template string's index
    within its category) so a downstream split can keep same-template rows
    together and avoid leaking near-duplicates across train/val/test.

Grounded in the same public sources as before: category names and
prevalence/loss statistics from the FBI IC3 2025 Internet Crime Report
(Elder Fraud section), plus "Grandparent Scam" and "Lottery/Sweepstakes",
both widely documented by AARP/BBB/FTC as elder-targeted categories (only
the category names/tactics are used -- no scraped text, per this project's
compliance notes: AARP and BBB Scam Tracker ToS were not touched).

All example text is original, written to reflect well-documented tactics
(urgency, authority impersonation, unusual payment methods, secrecy
requests) or, for the new legitimate_formal category, ordinary well-known
automated-notification phrasing (appointment reminders, shipping updates,
statement-ready notices) with no scraped or copied text.
"""
import csv
import itertools
import random

random.seed(42)

names = ["Grandma", "Grandpa", "Nana", "Pawpaw", "Mom", "Dad", "Robert", "Linda", "Uncle Jim", "Aunt Carol"]
amounts = ["$500", "$800", "$1,200", "$1,500", "$2,000", "$2,500", "$3,000", "$300", "$650", "$1,800"]
banks = ["Bank of America", "Wells Fargo", "Chase", "Citibank", "your credit union", "US Bank", "Capital One"]
companies = ["Microsoft", "Apple", "Amazon", "Norton", "Geek Squad", "PayPal", "HP Support", "Dell Support"]
agencies = ["the Social Security Administration", "the IRS", "Medicare", "the U.S. Marshals Service",
            "the Department of Treasury", "Medicaid", "the Department of Homeland Security"]
crypto = ["Bitcoin", "a Bitcoin ATM", "a cryptocurrency wallet", "USDT", "Ethereum"]
gift_cards = ["Amazon gift cards", "Google Play gift cards", "Target gift cards", "iTunes gift cards", "Steam gift cards"]
jackpots = ["$1,000,000", "$50,000", "$25,000", "$2,500,000", "$100,000"]
big_amounts = ["$10,000", "$15,000", "$20,000", "$12,500"]

# For the new legitimate_formal category
providers = ["Sharp HealthCare", "Kaiser Permanente", "Scripps Health", "UC San Diego Health",
             "your local clinic", "your primary care office"]
delivery_cos = ["USPS", "FedEx", "UPS", "your local post office"]
retailers = ["our store", "the pharmacy", "your neighborhood market", "the online shop"]
subscriptions = ["your streaming service", "your newspaper subscription", "your magazine subscription"]
doctors = ["Dr. Reynolds", "Dr. Patel", "Dr. Nguyen", "Dr. Alvarez", "Dr. Kim"]
ref_numbers = ["A-4471", "B-8825", "C-1093", "D-6642", "E-3357"]

examples = []
seen_texts = set()


def add_unique(text, scam_type, template_id, label=1, elder_targeted=1, source="synthetic_ic3_grounded"):
    text = text.strip()
    if text not in seen_texts:
        seen_texts.add(text)
        examples.append({
            "text": text, "label": label, "scam_type": scam_type,
            "elder_targeted": elder_targeted, "source": source,
            "template_id": f"{scam_type}::{template_id}",
        })


def fill_unique(template, template_id, var_lists, n, scam_type, label=1, elder_targeted=1, source="synthetic_ic3_grounded"):
    """var_lists: dict of placeholder_name -> list of options.
    Generates unique combos via cartesian product, shuffled, capped at n.
    All rows generated from the same `template` share one template_id, so a
    grouped train/val/test split can keep them together."""
    keys = list(var_lists.keys())
    combos = list(itertools.product(*[var_lists[k] for k in keys]))
    random.shuffle(combos)
    count = 0
    for combo in combos:
        if count >= n:
            break
        kwargs = dict(zip(keys, combo))
        text = template.format(**kwargs)
        before = len(examples)
        add_unique(text, scam_type, template_id, label, elder_targeted, source)
        if len(examples) > before:
            count += 1


# ============================== SCAM CATEGORIES ==============================

# --- Grandparent scam ---
grandparent_templates = [
    "Hi {name}, it's me, I'm in trouble. I was in a car accident and I'm at the police station. I need {amt} for bail but please don't tell Mom and Dad, I'm so embarrassed. Can you send it through a wire transfer today?",
    "{name}?? It's your grandson, I broke my phone so this is a friend's number. I'm stuck in Mexico, I need {amt} to get home. Please hurry and don't call my parents, I'll explain everything later.",
    "Grandma, please don't hang up. I'm in jail and I only get one call. My bail is set at {amt}. The lawyer said you can pay it with gift cards at the front desk. Please keep this between us.",
    "This is your granddaughter, I'm using a hospital phone. I was in an accident and need {amt} for the deductible before they'll treat me. Can you go to the pharmacy and send a MoneyGram right now?",
    "{name}, it's me, please don't be mad. I got pulled over and there was a warrant I didn't know about. The judge said I can pay {amt} to avoid going to court but I need it by tonight. Please don't tell anyone yet.",
    "Hi {name}, this is your grandson's lawyer. He asked me to call you directly. He was in a minor accident and needs {amt} for his bail to be released today. He's very embarrassed and asked that this stay between us for now.",
    "{name}? It's me, I lost my wallet and passport on my trip and the embassy needs {amt} to issue an emergency travel document. Can you send it today so I don't miss my flight home?",
    "Nana, it's me, I'm so sorry to ask this. My car broke down and the mechanic won't release it until I pay {amt}. Can you send it as gift cards? I'll pay you back as soon as I get home, please don't tell Mom.",
    "{name}, please listen carefully, I don't have much time on this phone. I was in a small accident and the other driver wants {amt} cash today or he's calling the police. Please don't tell Dad, I'll pay you back.",
    "It's me {name}, please don't hang up, I know my voice sounds different, I have a bad cold. I need {amt} wired today for a legal problem and I really don't want the rest of the family to worry about it.",
    "Hi {name}, this is the ER nurse calling on behalf of your grandchild. They asked me to reach you directly -- they need {amt} for a procedure not covered by insurance and asked that we keep this between us for now.",
    "{name}, I'm calling from the county jail, this call may be recorded. I was arrested last night, I need {amt} for bond and a legal fee. My public defender says gift cards work fastest, please hurry.",
    "It's your grandchild, please don't be upset. I'm overseas and got robbed, I lost everything including my passport. I need {amt} sent through {crypto} today so I can get an emergency flight home.",
]
for i, t in enumerate(grandparent_templates):
    fill_unique(t, i, {"name": names, "amt": amounts, "crypto": crypto}, 8, "grandparent_scam")

# --- Government impersonation (IRS/SSA/Medicare) ---
gov_templates = [
    "This is an automated message from {agency}. Your Social Security number has been suspended due to suspicious activity. Press 1 now to speak with an officer or your benefits will be permanently terminated.",
    "URGENT: {agency} has filed a lawsuit against you for unpaid taxes. To avoid arrest, you must call this number immediately and settle the balance of {amt} using gift cards.",
    "Your Medicare benefits are set to expire today. To keep your coverage active, please verify your Medicare number and date of birth by calling the number below within 24 hours.",
    "This is Officer Daniels with {agency}. A warrant has been issued for your arrest due to missed jury duty. You can resolve this by paying a {amt} processing fee over the phone right now.",
    "Attention: {agency} has detected an error on your latest tax filing. A refund of {amt} is pending, but you must confirm your bank account and Social Security number to release it.",
    "This is a final notice from {agency}. Failure to respond within 24 hours will result in suspension of your benefits and a referral to law enforcement. Call immediately to avoid legal action.",
    "Your case has been forwarded to {agency} for review due to unpaid fees of {amt}. To avoid a hold on your bank account, please call this number and settle the balance using a wire transfer.",
]
for i, t in enumerate(gov_templates):
    fill_unique(t, i, {"agency": agencies, "amt": amounts}, 8, "government_impersonation")

# --- Tech support scam ---
tech_templates = [
    "WARNING: Your computer has been infected with a virus and your personal files are at risk. Call {company} Support immediately at the number on your screen to prevent data loss. Do not restart your computer.",
    "This is {company} Technical Support. We detected unusual activity on your account from a device in another country. To secure your computer, please allow us remote access so we can remove the threat.",
    "Your antivirus subscription with {company} has renewed for {amt}. If you did not authorize this charge, call our billing department now to process a refund and we will need access to your bank account to verify.",
    "Hello, this is {company} calling about the ongoing security alert on your PC. We need you to purchase a gift card so we can charge a one-time protection fee and remove the malware we found.",
    "ALERT: {company} has flagged your computer for a critical security update. Your license has expired and your files will be locked in 24 hours unless you call the support line below.",
    "This is a courtesy call from {company}. Our servers show your computer was used to send spam from your account. To avoid a permanent ban, please give us remote access to run a free diagnostic.",
]
for i, t in enumerate(tech_templates):
    fill_unique(t, i, {"company": companies, "amt": amounts}, 8, "tech_support_scam")

# --- Romance scam ---
romance_templates = [
    "My darling, I think about you every day since we started talking. I finally got approved for the trip to see you, but customs is asking for a {amt} fee to release my luggage. Can you help me, my love?",
    "I have never felt this close to anyone online before. There's an emergency with my daughter overseas and the hospital needs {amt} before they will treat her. I promise I will pay you back the moment I land.",
    "Sweetheart, the oil rig contract finally came through and I'll be home soon, but I need {amt} to cover the shipping fee for my equipment. Please send it through {crypto} like we discussed.",
    "I've never trusted anyone the way I trust you. My bank account here is frozen because I'm military overseas. Could you send {amt} so I can book my flight home to finally meet you?",
    "My love, I was so excited to finally video call you but my phone was stolen along with my wallet. Could you send {amt} through {crypto} so I can replace it and we can talk again?",
    "Darling, I know we've only talked online, but I feel like I've known you my whole life. My shipping company needs {amt} in customs fees to release the gift I sent you. Can you help me clear it?",
    "My dearest, I was so looking forward to finally meeting you next week, but my company is holding my final paycheck until I pay a {amt} exit fee. Could you help so I don't miss my flight to you?",
    "I know you'll think I'm asking too much, but I have nowhere else to turn. My daughter is stranded at the airport and needs {amt} for a new ticket. I'll pay you back the moment my contract payment clears.",
    "You are the first person in years who has made me feel this way. I'm so close to being able to visit, but I need {amt} to clear an import tax on equipment for my work. Please, my love, I wouldn't ask if it wasn't urgent.",
    "My love, something terrible happened -- I was in a small accident overseas and the clinic won't release me until the {amt} bill is settled. Could you wire it through {crypto} today?",
]
for i, t in enumerate(romance_templates):
    fill_unique(t, i, {"amt": amounts, "crypto": crypto}, 8, "romance_scam")

# --- Investment fraud ---
investment_templates = [
    "Congratulations, you've been selected for our exclusive investment program guaranteeing 40% monthly returns with zero risk. Fund your account with {crypto} today to lock in this rate before it closes.",
    "Hi, this is Alex from the trading group you joined. Everyone in the group turned {amt} into {amt2} last week. Deposit now through our platform and I'll personally manage your portfolio.",
    "Your retirement savings deserve better returns. Our AI trading bot has a 98% win rate. Transfer {amt} today and watch your account grow daily. Limited spots available for new investors.",
    "As a valued member, you qualify for early access to a pre-IPO crypto opportunity. Minimum investment is {amt}, paid in {crypto}. Early investors are already seeing 3x returns.",
    "Your account has been pre-approved for our senior wealth-growth program. A one-time deposit of {amt} in {crypto} unlocks guaranteed weekly payouts starting immediately.",
    "This is your financial coach checking in -- the group's latest trade turned {amt} into {amt2} overnight. Don't miss the next opportunity, deposit today to join.",
]
for i, t in enumerate(investment_templates):
    fill_unique(t, i, {"amt": amounts, "amt2": big_amounts, "crypto": crypto}, 8, "investment_fraud")

# --- Business Email Compromise / gift card impersonation ---
bec_templates = [
    "Hi, are you available? I'm stuck in a meeting and need a favor. Can you pick up {gc} for a client gift and send me the codes? I'll pay you back at the office.",
    "Quick question -- I need you to purchase {gc} for our team appreciation event. Send me the redemption codes as soon as you get them, I'm heading into a call.",
    "I'm traveling and can't talk right now, but I need you to handle something urgent. Please buy {gc} and text me the numbers on the back. Keep this between us for now.",
    "Are you free? I need help with a time-sensitive task before end of day. Grab {gc} from any store nearby and send me a photo of the receipt and codes.",
    "Hey, sorry to bother you outside work hours, but I need a quick favor. Can you get {gc} for a surprise for the team? I'll reimburse you tomorrow, just need the codes tonight.",
    "This is urgent and confidential -- I need {gc} purchased right away for a vendor payment. Don't mention this to anyone else in the office until I confirm it went through.",
]
for i, t in enumerate(bec_templates):
    fill_unique(t, i, {"gc": gift_cards}, 6, "business_email_compromise")

# --- Lottery / sweepstakes scam ---
lottery_templates = [
    "CONGRATULATIONS! You have won {amt2} in the National Senior Sweepstakes. To claim your prize, please pay a small {amt} processing fee via wire transfer within 48 hours.",
    "This is an official notice: your name was drawn as the winner of our {amt2} giveaway. Call now to claim your prize -- a {amt} delivery and insurance fee applies before we can release your winnings.",
    "You've been selected to receive {amt2} from Publishers Sweepstakes! Just send {amt} in gift cards to cover the taxes and your check will be mailed immediately.",
    "Attention winner! Your ticket number matches our {amt2} jackpot drawing. To process your claim, a {amt} government release fee must be paid before funds can be transferred.",
    "You are the lucky recipient of a {amt2} prize from our annual senior rewards drawing. Send {amt} for processing and shipping to receive your winnings by courier this week.",
]
for i, t in enumerate(lottery_templates):
    fill_unique(t, i, {"amt": amounts, "amt2": jackpots}, 8, "lottery_sweepstakes")

# --- Phishing/spoofing ---
phishing_templates = [
    "Your {bank} account has been temporarily locked due to unusual sign-in activity. Verify your identity now by clicking the link below or your account will be closed within 24 hours.",
    "We were unable to process your recent {bank} payment. Please update your billing information immediately to avoid a suspension of your account.",
    "Your package could not be delivered due to an unpaid customs fee of $2.99. Click here to pay the fee and reschedule delivery before it is returned to sender.",
    "Security Alert: We noticed a new sign-in to your account from an unrecognized device. If this wasn't you, verify your password immediately using the secure link below.",
    "Your {bank} debit card has been flagged for suspicious purchases. Confirm your card number and PIN through the secure link below to prevent it from being closed.",
    "Final reminder: your {bank} online banking access will be suspended today unless you re-verify your account information through the link provided.",
]
for i, t in enumerate(phishing_templates):
    fill_unique(t, i, {"bank": banks}, 6, "phishing_spoofing")

# ========================== NEW: LEGITIMATE, FORMAL ==========================
# Original examples of ordinary automated notifications, written in formal
# business/institutional language but with none of the actual scam
# hallmarks: no payment demanded, no threat, no secrecy request, no request
# to click a link and "verify" personal/financial info. This directly
# targets the false-positive pattern found in real-world testing, where the
# dataset previously had zero examples of this style of legitimate message.

legit_formal_templates = [
    "Hello, there is an opening for a waitlisted appointment at {provider}. This offer is time sensitive and may be offered to other patients if we don't hear back. Please sign in to your account to review the available date.",
    "This is a reminder that your appointment with {doctor} is confirmed for next Tuesday at 10:00 AM. Please arrive 15 minutes early to complete any paperwork.",
    "Your recent lab results from {provider} are now available. Please log in to your patient portal to view them, or call our office if you have any questions.",
    "Your {bank} statement for this month is now available online. Log in through our official app or website to view your latest balance and transactions.",
    "Thank you -- your recent payment was received and processed successfully (reference {ref}). Your next billing cycle begins on the first of the month, as usual.",
    "Your order has shipped and is on its way! Track your package anytime through your account. Estimated delivery is in 3-5 business days via {delivery}.",
    "{delivery}: Your package is out for delivery today. No signature is required. You can track its progress using the tracking number in your account.",
    "This is a reminder that {subscription} will renew automatically next week. No action is needed if you'd like to continue -- log in to your account anytime to make changes.",
    "Your prescription refill is ready for pickup at {retailer} during normal business hours. Please bring a photo ID if this is your first pickup this year.",
    "This is a reminder from your library that a book you requested is ready for pickup. It will be held for 7 days before being returned to the shelf.",
    "Your reference number {ref} has been created for your recent service request. A representative will follow up within 2 business days if any further information is needed.",
    "Thank you for renewing your membership. Your new card will arrive by mail within 7-10 business days. No action is required on your part.",
    "This is a courtesy reminder that your annual wellness visit at {provider} is coming up. Please call our office at your convenience to schedule a time that works for you.",
    "Your utility payment was received, thank you. Your next billing statement will be available in your account on the usual date next month.",
    "We appreciate your feedback on your recent visit to {provider}. If you have a few minutes, a short survey link is available in your patient portal -- entirely optional.",
]
for i, t in enumerate(legit_formal_templates):
    fill_unique(
        t, i,
        {"provider": providers, "doctor": doctors, "bank": banks, "delivery": delivery_cos,
         "subscription": subscriptions, "retailer": retailers, "ref": ref_numbers},
        10, "legitimate_formal", label=0, elder_targeted=1, source="synthetic_legit_formal",
    )

# --- Legitimate messages that resemble scam-adjacent topics (hard negatives, label=0) ---
legit_templates = [
    "Hi Dad, just checking in -- how was your doctor's appointment today? Call me when you get a chance, love you.",
    "Your {bank} statement is now available online. Log in through our official app or website to view your latest balance.",
    "Reminder: your Medicare Part B premium payment is due on the 20th. No action is needed if you're enrolled in automatic withdrawal.",
    "This is a reminder from your pharmacy that your prescription refill is ready for pickup during normal business hours.",
    "Thanks for shopping with us! Your order has shipped and is expected to arrive in 3-5 business days. Track your package in your account.",
    "Hi Grandma, it's me! Just wanted to say happy birthday and that I'll call you this weekend. Miss you!",
    "Your appointment with Dr. Reynolds is confirmed for Tuesday at 10:00 AM. Please arrive 15 minutes early to complete paperwork.",
    "This is your library reminder: 'The Thursday Murder Club' is due back in 3 days. You can renew online or by phone.",
    "Your annual wellness visit is coming up. Please call our office at your convenience to schedule a time that works for you.",
    "Thank you for your donation to the local food bank. Your generosity helps feed families in our community this month.",
]
for i, t in enumerate(legit_templates):
    fill_unique(t, i, {"bank": banks}, 5, "legitimate", label=0)

if __name__ == "__main__":
    print(f"Total synthetic examples: {len(examples)}")
    scam_count = sum(1 for e in examples if e['label'] == 1)
    legit_count = sum(1 for e in examples if e['label'] == 0)
    print(f"Scam: {scam_count}, Legit: {legit_count}")
    print(f"Distinct template groups: {len(set(e['template_id'] for e in examples))}")

    with open("elder_synthetic_examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "scam_type", "elder_targeted", "source", "template_id"])
        writer.writeheader()
        for e in examples:
            writer.writerow(e)

    print("Saved elder_synthetic_examples.csv")
