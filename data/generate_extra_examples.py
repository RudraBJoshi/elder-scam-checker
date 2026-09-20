"""
Adds more diverse synthetic examples to the existing elder_focus_{train,val,test}.csv
splits, aimed at the two weakest areas found by error analysis and probing:

  1. legitimate_formal -- real institutional notices (clinics, banks, carriers,
     pharmacies, utilities, one-time codes, family texts) were being flagged as
     "medium" risk because only 15 templates of this style existed.
  2. Elder-targeted scams in styles the dataset barely covered: military
     romance, gift-card favor requests, fake subscription/"refund" callbacks,
     Medicare/benefit phishing, link-based smishing with lookalike domains.

Every row carries a template_id; whole templates are assigned to train/val/test
(about 84/8/8) so no template crosses a split boundary (same leakage rule as
rebuild_dataset.py). Safe to re-run: rows from this generator are identified by
their `source` and replaced, not duplicated.

Usage:  python data/generate_extra_examples.py
"""
import hashlib
import itertools
import pathlib
import random

import pandas as pd

HERE = pathlib.Path(__file__).parent
rng = random.Random(7)

SOURCE_LEGIT = "synthetic_legit_formal_v2"
SOURCE_SCAM = "synthetic_ic3_grounded_v2"

V = {
    "provider": ["Sharp HealthCare", "Kaiser Permanente", "Scripps Health", "Mercy Clinic", "Riverside Family Medicine",
                 "Oakwood Dental", "Lakeview Eye Care", "your pharmacy", "Walgreens", "CVS Pharmacy"],
    "doctor": ["Dr. Lee", "Dr. Shah", "Dr. Okafor", "Dr. Martinez", "Dr. Bennett", "Dr. Wu"],
    "bank": ["Chase", "Wells Fargo", "Bank of America", "US Bank", "Capital One", "your credit union", "Citibank"],
    "carrier": ["USPS", "FedEx", "UPS", "DHL"],
    "retailer": ["Amazon", "Walmart", "Target", "Costco", "Home Depot", "Best Buy"],
    "utility": ["San Diego Gas & Electric", "your water district", "your electric company", "Comcast", "Verizon", "AT&T"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "time": ["9:00 AM", "10:30 AM", "1:15 PM", "2:00 PM", "3:45 PM"],
    "last4": ["4421", "8830", "1057", "6392", "7714"],
    "amt_small": ["$12.40", "$38.75", "$52.10", "$84.20", "$19.99", "$127.63"],
    "code": ["483920", "771205", "392841", "605517", "218374"],
    "order": ["114-2233981", "112-8840217", "113-5521690", "111-9083374", "115-4407126"],
    "name": ["Grandma", "Grandpa", "Nana", "Pop", "Mom", "Dad"],
    "grandkid": ["Emily", "Jacob", "Sofia", "Daniel", "Maya", "Ben"],
    "amt": ["$500", "$800", "$1,200", "$2,000", "$2,500", "$3,000", "$450", "$1,750"],
    "agency": ["Social Security", "Medicare", "the IRS", "the Social Security Administration"],
    "brand": ["Amazon", "PayPal", "Norton", "McAfee", "Microsoft", "Geek Squad", "Apple"],
    "pay": ["Apple gift cards", "Google Play gift cards", "Target gift cards", "Bitcoin", "a wire transfer", "Zelle"],
    "country": ["Syria", "Afghanistan", "Nigeria", "Ghana", "Turkey", "the North Sea"],
    "domain_bad": ["secure-verify-login.top", "account-update.info", "portal-check.xyz", "myaccount-help.click",
                   "billing-support.co", "verify-now.online"],
    "good_url": ["https://photos.google.com/share/AF1Qip9x", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://www.allrecipes.com/recipe/23891",
                 "https://zoom.us/j/8842217755", "https://www.nytimes.com/2026/09/12/health", "https://www.facebook.com/events/551209",
                 "https://www.amazon.com/dp/B08N5WRWNW", "https://www.usps.com/manage", "https://www.chase.com/personal", "https://www.ssa.gov/myaccount",
                 "https://www.medicare.gov/plan-compare", "https://www.kaiserpermanente.org/appointments", "https://www.cvs.com/refill"],
    "svc": ["Google", "Microsoft", "Apple", "Amazon", "Netflix", "Spotify", "PayPal", "Facebook", "Zoom", "Dropbox", "Yahoo"],
    "app": ["Canva", "Duolingo", "Zoom", "Pinterest", "Airbnb", "Uber", "Slack", "a photo printing site", "a recipe app"],
    "month": ["June 3", "July 14", "August 9", "September 20", "October 2"],
    "device": ["a Windows computer", "an iPhone", "a Mac", "an Android phone", "a new browser"],
    "short": ["bit.ly/3xK9pQ", "tinyurl.com/y8k2mzp", "t.co/aB3dE9", "rb.gy/q7v1xz", "is.gd/Zk29Lm"],
}

rows = []


def fill(templates, scam_type, label, source, per_template=6):
    for ti, tpl in enumerate(templates):
        keys = [k for k in V if "{" + k + "}" in tpl]
        combos = list(itertools.product(*[V[k] for k in keys])) or [()]
        rng.shuffle(combos)
        seen = set()
        for combo in combos:
            text = tpl.format(**dict(zip(keys, combo)))
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "label": label, "scam_type": scam_type, "elder_targeted": 1,
                         "source": source, "template_id": f"v2::{scam_type}::{ti}"})
            if len(seen) >= per_template:
                break


# ================================ LEGITIMATE ================================
legit = [
    "Hi, this is {provider}. You have an appointment with {doctor} on {day} at {time}. Reply C to confirm or call the number on your appointment card to reschedule.",
    "Reminder: your appointment at {provider} is tomorrow at {time}. Please bring your insurance card and a list of your current medications.",
    "Your prescription is ready for pickup at {provider}. Reply STOP to opt out of these messages.",
    "{provider}: your refill for lisinopril is ready. Our pharmacy is open until 9 PM tonight. No payment is needed until pickup.",
    "{bank} alert: your card ending {last4} was used for {amt_small} at {retailer}. If this was you, no action is needed. If not, call the number on the back of your card.",
    "{bank}: a deposit of {amt_small} was posted to your account ending {last4}. View your balance in the official {bank} app.",
    "Your {bank} statement is ready. For your security, we will never ask for your password or PIN by text or email.",
    "Your verification code is {code}. It expires in 10 minutes. Do not share this code with anyone, including {bank} staff.",
    "{code} is your one-time passcode for {bank}. If you did not request it, you can ignore this message.",
    "Your {retailer} order #{order} has shipped and will arrive {day}. Track it anytime in the Your Orders section of the app.",
    "{retailer}: your order #{order} was delivered today. If something is wrong with your order, visit the Help section of the app or website.",
    "{carrier}: your package will be delivered {day}. No signature required. Track using the tracking number in your confirmation email at {carrier}.com.",
    "{carrier} update: your package is out for delivery today. You can follow its progress in the official {carrier} app.",
    "Your {utility} bill of {amt_small} is due on the 15th. You can pay through the official app, by mail, or set up autopay -- no action is needed if autopay is on.",
    "{utility}: your payment of {amt_small} was received. Thank you. Your next statement will be available on the 1st.",
    "Reminder from {utility}: scheduled maintenance is planned in your area on {day} between 10 AM and 2 PM. Service may be briefly interrupted.",
    "Hi {name}, just checking in. Are we still on for dinner on Sunday? Mom says to bring the pie. Love you!",
    "Hi {name}, it's {grandkid}. I got to school safe. I'll call you after practice. Love you lots!",
    "Hey {name}, can you pick me up at 5? My car is at the shop. Thanks so much!",
    "{name}, happy birthday! We're all coming over on Saturday with cake. Let us know if you need anything from the store.",
    "This is {provider}. Your lab results are available in your patient portal. Log in at the address on your visit summary, or call our office with questions.",
    "Your annual Medicare open enrollment period begins soon. Visit medicare.gov or call 1-800-MEDICARE to review your options. We will never call you to ask for your Medicare number.",
    "Your Social Security statement is available. Sign in to your my Social Security account at ssa.gov to view it. We will never ask for payment by gift card.",
    "Your library hold is ready for pickup at the front desk and will be kept for 7 days.",
    "Thank you for your order. Your receipt for {amt_small} is attached. Questions? Reply to this email or visit our store during business hours.",
    "Weather alert: freezing temperatures are expected tonight. Bring pets indoors and protect exposed pipes. This is an automated message from your county emergency service.",
    "Your flight to Phoenix departs {day} at {time}. Check in opens 24 hours before departure on the airline's official app.",
    "Reminder: your dental cleaning at {provider} is on {day} at {time}. Please call our office if you need to reschedule.",
    "{bank}: we noticed you have not used your card ending {last4} recently. It is still active. No action is needed.",
    "Your subscription will renew on the 1st for {amt_small}. Manage or cancel anytime in your account settings. No action is needed to continue.",
    "The pharmacy closes early today at 6 PM for the holiday. Your prescriptions will be available tomorrow morning.",
    "Your neighborhood watch meeting is {day} at {time} at the community center. Coffee and donuts provided. Hope to see you there.",
    "Your car is ready for pickup at the service center. The total is {amt_small}, payable when you arrive. Thank you for choosing us.",
]
legit += [
    "Hi {name}, here are the photos from the trip: {good_url} Let me know which ones you want printed!",
    "{name}, look at this recipe I mentioned, I think you'll love it: {good_url}",
    "Hey {name}, the family video call is {day} at {time}. Here's the link: {good_url} Just tap it to join.",
    "Saw this article about your favorite garden club and thought of you: {good_url}",
    "Here is the link to the event page for the church picnic: {good_url} See you there!",
    "{provider}: you can view your visit summary and message your care team anytime at {good_url}",
    "{bank}: to view your latest statement, sign in to your account directly at {good_url} We will never ask for your password by text.",
    "Your {carrier} package is scheduled for {day}. You can track it at {good_url} using the tracking number in your receipt.",
    "Your prescription refill is ready. Manage refills at {good_url} or call the pharmacy during business hours.",
    "Open enrollment reminder: compare plans at {good_url} or call the number on the back of your card. We will never ask you to pay by gift card.",
    "Hi {name}, I sent you the tickets for {day}. You can also find them at {good_url}",
    "Just a reminder that your book club is meeting {day} at {time}. Directions and the reading list are at {good_url}",
]
legit += [
    "Keep track of your {svc} Account data. You're receiving this email because you used Sign in with {svc} to sign in to {app} on {month} at {time}. This email summarizes the info you shared. There is nothing you need to do right now.",
    "Security alert: new sign-in to your {svc} Account from {device} on {month}. If this was you, you don't need to do anything. If not, you can review recent activity in your account settings.",
    "{svc}: your password was changed successfully on {month}. If you made this change, no action is needed. Questions? Visit the {svc} help center.",
    "Welcome to {svc}! Your account has been created. You're receiving this email because you signed up on {month}. You can unsubscribe or change your email settings at any time.",
    "Thanks for your purchase. Your receipt from {svc} for {amt_small} is attached. You're receiving this email because you made a purchase on {month}. No action is required.",
    "Your {svc} ID was used to sign in on {device}. If this was you, no action is needed. This is an automated message, please do not reply.",
    "{svc} 2-step verification is now on for your account. From now on you'll get a prompt on your phone when you sign in. You don't need to do anything else.",
    "Your monthly {svc} summary is ready: you used 12 GB of storage this month. Manage your plan any time in your account settings. No action is needed.",
    "You're receiving this email because you have an account with {svc}. We've updated our privacy policy effective {month}. You can read the changes in your account settings. There's nothing you need to do.",
    "Hi, your {svc} subscription receipt for {amt_small} is below. Billing period: {month}. To manage or cancel, open the {svc} app and go to Settings. Thank you for being a member.",
    "{svc}: you signed in to {app} using your {svc} account on {month}. The info shared with {app} includes your name and email address. You can review or remove access anytime in your account settings.",
    "Your {svc} storage is almost full. You've used 90% of your free space. You can free up space or view upgrade options in settings whenever you're ready.",
]
fill(legit, "legitimate_formal", 0, SOURCE_LEGIT, per_template=8)

# ================================== SCAMS ===================================
scams = [
    # link-based smishing
    "{carrier}: Your package is on hold due to an unpaid $1.95 delivery fee. Pay now at {domain_bad}/pay to reschedule or it will be returned.",
    "{bank} Alert: unusual sign-in detected on your account. Confirm your identity now: {short}",
    "Your {brand} account has been locked. Verify your information within 24 hours at {domain_bad} or your account will be closed.",
    "Congratulations! You have been selected for a $1000 {retailer} gift card. Claim your reward before midnight: {short}",
    "{agency} notice: a payment is pending in your name. Confirm your identity at {domain_bad} to receive it.",
    "Your Medicare card is being replaced. Confirm your name, date of birth and Medicare number at {domain_bad} to avoid losing coverage.",
    # subscription / refund callback
    "Your {brand} subscription has renewed for $399.99. If you did not authorize this charge, call 1-855-555-0{last4} right away for a full refund.",
    "Invoice: your {brand} protection plan was charged $479.00. To cancel and receive your refund, call our billing department today at 1-888-555-0{last4}.",
    "We have issued you a refund of $299 by mistake. Please call us at 1-844-555-0{last4} so we can correct it before your account is charged a penalty.",
    # tech support
    "WARNING: your computer is infected with a virus. Call {brand} Support now at 1-800-555-0{last4}. Do not restart or shut down your computer.",
    "{brand} Security Alert: your device has been compromised and your files are at risk. Call the toll-free number immediately for a technician to remote in and fix it.",
    # romance / military
    "Hello dear, I am a US soldier serving in {country}. I have fallen in love with your profile. I need {amt} for leave papers to visit you. Please send it as {pay}.",
    "My darling, I am on an oil rig and cannot access my bank. I need {amt} for a customs fee to bring my gift to you. You are the only one I trust. Please send {pay} and keep this between us.",
    "I know we only met online but I feel so close to you. My daughter is sick and I need {amt} for her hospital bill. I will pay you back when I return. Please use {pay}.",
    # gift card favor / boss
    "Hi, it's your pastor. I am stuck in a meeting and can't talk right now. Can you quietly pick up {pay} for a member who is ill? I'll reimburse you. Please keep this between us.",
    "Are you available? I need a quick favor. Please buy {pay} worth {amt} and send me the codes. I can't take calls right now. Don't mention this to anyone.",
    # grandparent variants
    "{name}, it's {grandkid}. I'm in trouble and I lost my phone, this is a new number. I need {amt} sent by {pay} today. Please don't tell Mom and Dad.",
    "Grandma this is {grandkid}, I was arrested after an accident and I need {amt} for bail. The lawyer will call you. Please don't tell anyone, I'm so embarrassed. Send {pay}.",
    # government / benefits
    "This is {agency}. Your account has been suspended due to suspicious activity and a warrant may be issued. Call 1-888-555-0{last4} immediately to resolve this.",
    "Notice from {agency}: you are owed a benefit increase. To receive it you must confirm your Social Security number and bank account number by replying to this message.",
    "Your Social Security payment will be stopped unless you verify your identity today. Call our agent at 1-877-555-0{last4}. Do not discuss this with anyone at your bank.",
    # investment
    "I made $8,000 last week with a crypto trading group. Guaranteed returns with zero risk. Send {amt} in Bitcoin and I'll show you how to double it in 48 hours.",
    # lottery
    "You have won a $500,000 sweepstakes! To release your prize, pay a processing fee of {amt} using {pay}. Do not tell anyone until your prize is delivered.",
    # job
    "Work from home opportunity! Earn $500 daily with no experience necessary. Reply with your full name, address, and bank details to receive your paid trial.",
    # utility threat
    "{utility}: your power will be shut off in 1 hour unless you pay {amt} now. Pay with {pay} at the number below to avoid disconnection.",
]
fill(scams, "scam_v2", 1, SOURCE_SCAM, per_template=7)


def main():
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    # Stable per-template assignment (hash, not shuffle) so adding templates never
    # moves existing ones between splits.
    def bucket(tid):
        h = int(hashlib.md5(tid.encode()).hexdigest(), 16) % 100
        return "test" if h < 8 else "val" if h < 16 else "train"

    split = df["template_id"].map(bucket)

    for name in ["train", "val", "test"]:
        path = HERE / f"elder_focus_{name}.csv"
        cur = pd.read_csv(path)
        cur = cur[~cur["source"].isin([SOURCE_LEGIT, SOURCE_SCAM])]
        add = df[split == name]
        out = pd.concat([cur, add], ignore_index=True)
        out.to_csv(path, index=False)
        print(f"{name}: +{len(add)} rows ({int(add['label'].sum())} scam), now {len(out)}")


if __name__ == "__main__":
    main()
