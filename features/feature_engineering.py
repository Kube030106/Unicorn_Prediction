import numpy as np
import pandas as pd
from datetime import datetime

CURRENT_YEAR = datetime.now().year

INDUSTRY_RISK = {
    "SaaS": 0.9,
    "FinTech": 0.85,
    "HealthTech": 0.8,
    "EdTech": 0.7
}

TOP_COUNTRIES = ["USA", "India", "UK"]

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Startup Age
    df["startup_age"] = CURRENT_YEAR - df["founded_year"]

    # Avoid division by zero
    df["startup_age"] = df["startup_age"].replace(0, 1)

    # 2. Funding Speed
    df["funding_speed"] = df["total_funding"] / df["startup_age"]

    # 3. Funding Intensity
    df["funding_intensity"] = df["total_funding"] / df["funding_rounds"]

    # 4. Early-stage funding ratio
    df["early_funding_ratio"] = df["funding_first_2_years"] / df["total_funding"]

    # 5. Funding continuity
    df["funding_continuity"] = (df["funding_rounds"] >= 2).astype(int)

    # 6. Investor density
    df["investor_density"] = df["investor_count"] / df["funding_rounds"]

    # 7. Industry risk score
    df["industry_risk"] = df["industry"].map(INDUSTRY_RISK).fillna(0.6)

    # 8. Geography strength
    df["geo_strength"] = df["country"].apply(
        lambda x: 1.0 if x in TOP_COUNTRIES else 0.5
    )

    # 9. Growth score (composite)
    df["growth_score"] = (
        0.4 * df["funding_speed"] +
        0.3 * df["investor_density"] +
        0.3 * df["early_funding_ratio"]
    )

    # 10. Valuation proxy
    df["valuation_proxy"] = np.log1p(df["total_funding"])

    return df