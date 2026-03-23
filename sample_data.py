"""
sample_data.py — Generates a realistic synthetic CSV of cognitive test sessions.

Structure: 20 users × 2-4 sessions each ≈ 50 rows total.
Three latent groups ensure KMeans finds clean clusters visually:
  - Focused : fast + accurate
  - Fatigued : slow + inaccurate
  - Impulsive: fast + inaccurate + high error
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_data(seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of synthetic cognitive test sessions."""
    rng = np.random.default_rng(seed)

    test_types = ["Stroop", "N-Back", "Go/No-Go", "Trail-Making", "Flanker"]

    # ------------------------------------------------------------------ #
    # Profile definitions: (mean_rt, std_rt, mean_acc, std_acc,           #
    #                        mean_err, std_err, n_users)                   #
    # ------------------------------------------------------------------ #
    profiles = {
        "Focused": {
            "rt_mean": 245, "rt_std": 30,
            "acc_mean": 94, "acc_std": 3,
            "err_mean": 4,  "err_std": 1.5,
            "n_users": 7,
        },
        "Fatigued": {
            "rt_mean": 440, "rt_std": 50,
            "acc_mean": 67, "acc_std": 5,
            "err_mean": 19, "err_std": 4,
            "n_users": 7,
        },
        "Impulsive": {
            "rt_mean": 210, "rt_std": 25,
            "acc_mean": 71, "acc_std": 6,
            "err_mean": 24, "err_std": 5,
            "n_users": 6,
        },
    }

    rows = []
    user_counter = 1
    base_date = datetime(2024, 1, 15)

    for profile_name, cfg in profiles.items():
        for _ in range(cfg["n_users"]):
            uid = f"user_{user_counter:03d}"
            user_counter += 1
            n_sessions = rng.integers(2, 5)  # 2–4 sessions per user

            for s in range(n_sessions):
                # Add slight per-session noise to simulate progress/decline
                drift = rng.uniform(-0.05, 0.05)

                rt = float(rng.normal(
                    cfg["rt_mean"] * (1 + drift * 0.5),
                    cfg["rt_std"]
                ))
                acc = float(np.clip(rng.normal(
                    cfg["acc_mean"] * (1 - drift * 0.3),
                    cfg["acc_std"]
                ), 0, 100))
                err = float(np.clip(rng.normal(
                    cfg["err_mean"] * (1 + drift * 0.4),
                    cfg["err_std"]
                ), 0, 100))
                n_trials = int(rng.integers(30, 80))
                test_type = rng.choice(test_types)
                session_date = base_date + timedelta(days=int(s * 7 + rng.integers(0, 3)))

                rows.append({
                    "user_id": uid,
                    "session_number": s + 1,
                    "session_date": session_date.strftime("%Y-%m-%d"),
                    "reaction_time_ms": round(max(rt, 100), 1),
                    "accuracy_pct": round(min(max(acc, 0), 100), 1),
                    "error_rate": round(min(max(err, 0), 100), 1),
                    "n_trials": n_trials,
                    "test_type": test_type,
                    # Hidden ground truth — useful for debugging but ignored by model
                    "_true_profile": profile_name,
                })

    df = pd.DataFrame(rows).sort_values(["user_id", "session_number"]).reset_index(drop=True)
    return df


def get_user_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multi-session data to one row per user.
    Uses the mean across all sessions — this is the input to KMeans.
    """
    agg = (
        df.groupby("user_id")
        .agg(
            reaction_time_ms=("reaction_time_ms", "mean"),
            accuracy_pct=("accuracy_pct", "mean"),
            error_rate=("error_rate", "mean"),
            n_trials=("n_trials", "mean"),
            n_sessions=("session_number", "max"),
        )
        .reset_index()
    )
    agg["reaction_time_ms"] = agg["reaction_time_ms"].round(1)
    agg["accuracy_pct"] = agg["accuracy_pct"].round(1)
    agg["error_rate"] = agg["error_rate"].round(1)
    agg["n_trials"] = agg["n_trials"].round(0).astype(int)
    return agg


if __name__ == "__main__":
    df = generate_sample_data()
    print(f"Generated {len(df)} rows for {df['user_id'].nunique()} users")
    print(df.head(10).to_string(index=False))
    df.to_csv("sample_cognitive_data.csv", index=False)
    print("\nSaved to sample_cognitive_data.csv")
