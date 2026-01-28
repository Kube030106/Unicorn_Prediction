🧠 What features are ACTUALLY driving the prediction?

Your model predicts startup success (IPO / Acquired vs Others) using signals of funding strength, investor confidence, traction, and ecosystem quality.

Below are the main feature groups, in plain English.

1️⃣ Funding Strength (MOST IMPORTANT)

These features tell the model how much belief investors have shown.

🔹 funding_total_usd

Total capital raised

Strongest proxy for scale & traction

Higher → much higher success probability

🔹 funding_rounds

Number of funding events

Multiple rounds = sustained investor confidence

📌 These two usually carry the highest importance in tree-based models.

2️⃣ Growth Timeline (Speed & Momentum)

These tell how fast the startup progressed.

🔹 age_first_funding_year

Time taken to raise first funding

Faster = strong early validation

🔹 age_last_funding_year

Time between founding and latest funding

Shorter gap = rapid growth

🔹 milestones

Number of achieved milestones

Indicates product / market progress

3️⃣ Investor Quality & Network Strength

These capture who is backing the startup.

🔹 has_VC

Presence of venture capital firms

VC backing dramatically increases success odds

🔹 has_angel

Early smart money signal

🔹 avg_participants

Average investors per round

More participants = broader confidence

🔹 relationships

Size of professional network

Strong ecosystem support

4️⃣ Funding Stage Progression (Trajectory Signal)

These show whether the startup advanced through funding stages.

🔹 has_roundA
🔹 has_roundB
🔹 has_roundC
🔹 has_roundD

Progressing beyond Round B is a very strong success indicator.

5️⃣ Geography Strength (Ecosystem Effect)

Location matters in startup success.

🔹 is_CA (California / Silicon Valley)
🔹 is_NY
🔹 is_MA
🔹 is_TX

Top ecosystems provide:

Better talent

Better investors

Faster exits

6️⃣ Industry / Category Advantage

Different industries have different success odds.

🔹 is_software
🔹 is_web
🔹 is_mobile
🔹 is_enterprise
🔹 is_ecommerce
🔹 is_biotech
🔹 is_consulting

Software, enterprise, and biotech usually outperform consulting or niche sectors.

🎯 TL;DR — Top Predictive Signals

If you had to explain it in one sentence:

The model mainly looks at how much money the startup raised, how fast it raised it, who invested, whether it progressed through funding stages, and whether it operates in a strong ecosystem and industry.

📌 Interview-Ready Summary (USE THIS)

Our model predicts startup success using funding magnitude and continuity, investor quality signals, growth velocity, funding stage progression, and ecosystem strength, which together act as strong proxies for scalability and market validation.