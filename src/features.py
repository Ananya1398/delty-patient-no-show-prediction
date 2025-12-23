"""
Please refer to notebooks/delty.ipynb for a complete explanation of
feature selection, sensitivity, and visualizations.
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """
    Build features for both training and inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (raw training data OR single-row inference data)
    training : bool
        If True, create target variable `no_show`
        If False, skip target creation (used during inference)

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Target (Only during training)
    # ------------------------------------------------------------------
    if training and "No-show" in df.columns:
        df["no_show"] = (
            df["No-show"].str.strip().str.lower() == "yes"
        ).astype(int)

    # ------------------------------------------------------------------
    # Datetime parsing
    # ------------------------------------------------------------------
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True, errors="coerce")
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True, errors="coerce")

    # ------------------------------------------------------------------
    # Lead time (DATE-based to avoid midnight artifacts)
    # ------------------------------------------------------------------
    sched_date = df["ScheduledDay"].dt.date
    appt_date = df["AppointmentDay"].dt.date
    df["lead_time_days"] = (
        pd.to_datetime(appt_date) - pd.to_datetime(sched_date)
    ).dt.days

    # ------------------------------------------------------------------
    # Cleaning (safe for both training & inference)
    # ------------------------------------------------------------------
    if training:
        df = df[(df["Age"] >= 0) & (df["Age"] <= 110)]
        df = df[df["lead_time_days"] >= 0]

    # ------------------------------------------------------------------
    # Time-based features
    # ------------------------------------------------------------------
    df["is_same_day"] = (df["lead_time_days"] == 0).astype(int)
    df["appt_dow"] = df["AppointmentDay"].dt.dayofweek
    df["appt_month"] = df["AppointmentDay"].dt.month
    df["scheduled_dow"] = df["ScheduledDay"].dt.dayofweek
    df["scheduled_hour"] = df["ScheduledDay"].dt.hour

    # ------------------------------------------------------------------
    # Health-related features
    # ------------------------------------------------------------------
    df["has_chronic"] = (
        (df["Hypertension"] == 1) | (df["Diabetes"] == 1)
    ).astype(int)

    df["comorbidity_count"] = (
        df["Hypertension"] + df["Diabetes"] + df["Alcoholism"]
    )

    df["has_disability"] = (df["Handicapped"] > 0).astype(int)

    # ------------------------------------------------------------------
    # Drop leakage / unused columns (if present)
    # ------------------------------------------------------------------
    drop_cols = ["PatientId", "AppointmentID", "No-show"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df
