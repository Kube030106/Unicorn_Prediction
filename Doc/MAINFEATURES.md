🧠 Features That ACTUALLY Change the Output (High → Low Impact)

Based on:

Your dataset (Crunchbase-style)

Your models (XGBoost + RF + GB)

Typical feature importance patterns for startup success

🥇 TIER 1 — STRONGEST IMPACT FEATURES

(Small change → big probability shift)

🔹 funding_total_usd

Most influential feature

Increasing funding → sharply increases success probability

📌 Example
$5M → $50M can move probability from 0.3 → 0.7

🔹 funding_rounds

More rounds = sustained confidence

Jump from 1–2 rounds → 4–5 rounds = huge boost

🔹 has_roundB, has_roundC, has_roundD

These are stage-gates

Crossing Round B is often a tipping point

📌 Even one of these flipping from 0 → 1 can change outcome.

🔹 has_VC

VC backing is a strong prior for success

Stronger than angel-only funding

🥈 TIER 2 — HIGH IMPACT (Contextual Boosters)
🔹 age_first_funding_year

Faster early funding → better signal

Decreasing this improves prediction

🔹 age_last_funding_year

Shorter time to last funding = higher momentum

🔹 milestones

Product / market progress

Especially important for GB & XGBoost

🔹 avg_participants

More investors per round → stronger validation

🥉 TIER 3 — MODERATE IMPACT (Directional)
🔹 relationships

Network strength

Amplifies funding signals

🔹 is_software, is_enterprise, is_biotech

Category advantage

Software & biotech usually positive

🔹 is_CA, is_NY, is_MA

Ecosystem effect

Helps when combined with funding strength

🔻 LOW IMPACT (Rarely Change Output Alone)

These usually don’t flip predictions alone:

is_consulting

is_web

is_mobile (unless combined)

is_TX

has_angel (without VC)

is_ecommerce (mixed effect)

🎯 WHICH FEATURES TO CHANGE TO FLIP A PREDICTION
❌ Low probability → ✅ High probability

Change at least one Tier-1 feature:

✔ Increase funding_total_usd
✔ Increase funding_rounds
✔ Flip has_roundB from 0 → 1
✔ Flip has_VC from 0 → 1

Example Scenario
Feature	Before	After
funding_total_usd	3M	25M
funding_rounds	1	4
has_roundB	0	1
has_VC	0	1

📈 Probability jumps ~0.25 → ~0.8

🧠 WHY SOME FEATURES DON’T MATTER MUCH

Because:

Trees already capture their effect via stronger features

They act as secondary conditioners, not drivers

They matter only in combination