# Report: Patient No-Show Prediction (Baseline)

## Summary
I built a baseline no-show prediction pipeline using an 80/20 stratified train/test split on 110,516 cleaned appointments. I trained two models: (1) Logistic Regression as a simple baseline and (2) XGBoost Gradient Boosting as the primary model. The no-show rate is ~20.2%, so evaluation focuses on both ranking quality (AUC-ROC) and operational decision-making (prioritizing outreach calls).

## Data Quality + Feature Engineering (high-level)
- **No missing values**, but temporal fields required cleaning: `AppointmentDay` is date-like (00:00:00Z) while `ScheduledDay` includes time-of-day.
- I computed **lead_time_days** using **date-level difference** and dropped a small number of invalid rows (Age > 110, and a few negative lead times). Total removed: 11 rows.
- Key engineered features (leakage-safe, known at scheduling time):  
  `lead_time_days`, `is_same_day`, `appt_dow`, `scheduled_dow`, `scheduled_hour`, `appt_month`, plus composite health features (`comorbidity_count`, `has_chronic`, `has_disability`).  
  Categorical fields (`Gender`, `Neighbourhood`) were one-hot encoded. Identifiers (`PatientId`, `AppointmentID`) were excluded to avoid memorization.

## Metrics (Test Set)
### Logistic Regression (threshold = 0.50)
- **AUC-ROC:** 0.7206  
- **Precision:** 0.3001  
- **Recall:** 0.8541  
- **F1:** 0.4442  

### XGBoost (threshold = 0.50)
- **AUC-ROC:** 0.7411  
- **Precision:** 0.5391  
- **Recall:** 0.0294  
- **F1:** 0.0557  

**Note on thresholds:** XGBoost’s probabilities are not well-calibrated at the default 0.50 cutoff, leading it to predict “no-show” rarely. When selecting a threshold to optimize F1, Logistic Regression performs best around **0.53 (F1≈0.445)** and XGBoost around **0.18–0.21 (F1≈0.454)**. AUC-ROC and top-K prioritization are more informative for operational use than a fixed 0.50 threshold. (described in the section below)
### Top-K evaluation aligned to clinic capacity (K = 100 calls/day)
A fixed threshold (like 0.50) is not always the best operational choice when the clinic has a hard outreach limit. For the “100 calls/day” scenario, a more relevant metric is **Precision@100** (“of the 100 called, what fraction are true no-shows?”):

- **Baseline no-show rate:** 20.19%  
- **Logistic Regression Precision@100:** 39% (39/100 no-shows captured)  
- **XGBoost Precision@100:** 59% (59/100 no-shows captured)  

This shows XGBoost is substantially better for prioritizing limited outreach.
## Which model performed better? By how much?
- **XGBoost performed better overall** on ranking quality: **AUC-ROC 0.7411 vs 0.7206** (Δ ≈ **+0.0205**).
- Under the operational constraint of limited outreach capacity (top-K calling), XGBoost also provided substantially higher yield (see the 100 calls/day section).

## Why do I think that model performed better?
XGBoost can learn **non-linear relationships and feature interactions** (e.g., how lead time interacts with same-day scheduling patterns and neighborhood effects) that a linear model cannot capture. This improves ranking and prioritization even when probabilities are not perfectly calibrated at a default threshold.

## Which features were most important?
Based on XGBoost feature importance, the strongest drivers were:
- **is_same_day** (largest signal)
- **lead_time_days**
- **SMS_received** (likely influenced by outreach policy/selection effects)
- **Neighbourhood** indicators (proxy for access/transport constraints)
- **Age**, **OnGovtWelfareBenefits**, and other health/disability flags

These align with exploratory analysis: same-day appointments had far lower no-show rates, and longer lead times showed increasing no-show rates across buckets.

## Tradeoffs between precision and recall
- **High recall** (catching most no-shows) typically requires a **lower threshold**, but increases false positives (more unnecessary calls).
- **High precision** (fewer false positives) requires a **higher threshold**, but misses many no-shows.
- The “best” operating point depends on the clinic’s capacity and the cost of outreach vs. the cost of a missed appointment.

## If the clinic can only make 100 outreach calls per day, how to prioritize?
Use the model as a **ranking system**:
1. Compute predicted no-show probability for each scheduled appointment.
2. Sort descending by risk.
3. Call the **top 100 highest-risk** patients.

This directly optimizes outreach efficiency:
- Baseline no-show rate: **~20.19%**  
- **Top 100** by Logistic Regression: **39%** no-shows (39/100)  
- **Top 100** by XGBoost: **59%** no-shows (59/100)  

So XGBoost yields ~**3×** the no-show capture compared to random selection and significantly outperforms Logistic Regression for prioritization.

## Limitations and what I would improve with more time
- The dataset lacks richer context (e.g., appointment type, provider, distance, prior attendance history). These are likely important drivers of no-shows.
- I intentionally avoided patient history features to prevent leakage; with more time, I would implement **time-aware historical features** (e.g., prior no-show rate) using strict chronological splits.
- I would run **cross-validation** and light hyperparameter tuning, and add **probability calibration** 
- I would also evaluate fairness/bias and subgroup performance (e.g., across neighborhoods or socioeconomic proxies) to ensure the model does not amplify disparities.
- I would deploy on GCP cloudrun, build a streamlit/ react frontend and host models on Vertex AI
