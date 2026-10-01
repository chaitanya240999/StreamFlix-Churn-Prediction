StreamFlix Customer Churn Prediction & Segmentation
Predicting which streaming subscribers will cancel within 30 days, and grouping customers into segments a retention team can act on.

Results at a glance

ROC-AUC 0.75 on a held-out test set (tuned gradient boosting).
The 10% of customers the model ranks riskiest churn at 44.5%, or 2.35× the 18.9% base rate.
Customer segmentation found a low-engagement segment churning at 34% (1.8× average) and a payment-failure segment churning at 25%.
Problem
StreamFlix loses revenue every time a subscriber cancels. With an 18.9% churn rate across 10,000 subscribers, even a small improvement in retention is worth a lot. The goal is to score every customer's churn risk early enough for the retention team to step in, and to explain why customers leave.

Target: will_churn = 1 if the customer cancelled within 30 days.
Data: 10,000 subscribers and 37 features covering subscription terms, viewing habits, content preferences, devices, engagement and support history. The dataset was provided for DATA 602 at UMBC.
Approach
Why accuracy is the wrong metric
81.1% of customers don't churn, so a model that always predicts "no churn" is 81.1% accurate and completely useless. Models are selected on ROC-AUC, which measures how well the model ranks churners above non-churners, and judged on churn-class recall and precision at a chosen threshold.

Feature engineering
Nine features combine raw signals into something a business user can read:

Feature	Definition	Why it matters
engagement_score	Watch hours × completion rate × sessions	One number for how actively someone uses the platform
inactivity_flag	No viewing for more than 7 days	Recent inactivity is a leading churn signal
price_per_day	Monthly price ÷ 30	Normalized cost perception
high_support_user	2+ support tickets	Unresolved friction drives disengagement
abandonment_rate	Abandoned series ÷ unique titles watched	Unfinished content signals waning interest
multi_device_user	3+ devices	Cross-device users are more invested
payment_risk	Payment failures + account on hold	Combined billing friction
subscription_start_year, subscription_start_month	From the start date	When the customer joined
High-cardinality text columns (genres_watched, devices_used, support_ticket_reasons) were dropped because their structured counterparts already carry the signal.

Modeling
Every model is a scikit-learn Pipeline that bundles imputation, scaling and one-hot encoding with the model, so test data never leaks into preprocessing.
Stratified 80/20 train/test split. The test set is untouched until final evaluation.
Four models compared with stratified 5-fold cross-validation; the winner was tuned with GridSearchCV.
Model	CV ROC-AUC (mean ± std)
Logistic Regression	0.718 ± 0.018
Random Forest	0.744 ± 0.017
Gradient Boosting	0.762 ± 0.016
AdaBoost	0.761 ± 0.017
Tuned gradient boosting: learning_rate=0.05, max_depth=3, n_estimators=100 (CV ROC-AUC 0.763).

Results (held-out test set)
Ranking quality
Model	ROC-AUC	Churn recall @ 0.5	Churn precision @ 0.5	Accuracy
Logistic Regression	0.708	0.648	0.316	0.668
Random Forest	0.738	0.037	0.538	0.812
Gradient Boosting (tuned)	0.749	0.077	0.537	0.813
AdaBoost	0.752	0.140	0.457	0.806
The model was chosen on cross-validation, not on test scores. AdaBoost's slightly higher test AUC (0.752 vs 0.749) is within the cross-validation noise (±0.016).

Choosing a threshold
At the default 0.5 cut-off the model flags only 2.7% of customers and catches 7.7% of churners. That cut-off is wrong for an 18.9% base rate. The right one depends on how many customers the retention team can contact:

Threshold	Churn recall	Churn precision	Customers flagged
0.50	7.7%	53.7%	2.7%
0.40	20.1%	47.2%	8.1%
0.30	44.2%	39.6%	21.1%
0.25	54.5%	37.7%	27.3%
0.20	64.8%	35.2%	34.8%
0.15	76.2%	30.7%	46.9%
Lift
Customers targeted	Churn rate	Lift vs. 18.9% base	Share of all churners caught
Top 10% by risk score	44.5%	2.35×	23.5%
Top 20% by risk score	40.8%	2.16×	43.1%
What drives churn
The top features by importance: engagement_score, days_since_last_watch, subscription_length_days, total_watch_time_hours, sessions_count, account_on_hold and payment_risk. Customers who stop watching, and customers with payment problems, are the ones who leave.

Customer segmentation (unsupervised)
churn_segmentation.py clusters customers with K-Means on behavior only. The churn label is held out and used afterwards only to profile each segment.

First attempt failed: one cluster held 91% of customers and another held only 40 extreme outliers, because usage counts are heavily skewed (support tickets had a skew of about 8).
Fix: log-transform skewed counts and turn rare events (payment failures, support tickets) into yes/no flags. k = 4 was chosen for silhouette score and for keeping every segment at 14% or more of customers.
Segment	Share	Churn rate	Defining behavior
Low engagement	15%	34% (1.8×)	Median 0 sessions, ~2 watch hours
Payment failures	14%	25% (1.35×)	Every customer had a failed payment
Moderate users	46%	17%	~10 watch hours
Heavy users	25%	10%	~47 watch hours
Recommended actions: re-engagement campaigns for the low-engagement segment (personalized picks, "continue watching" reminders), and payment recovery for the payment-failure segment (retry logic, card-update prompts) before cancellation.

Limitations
ROC-AUC of 0.75 is a useful ranking but not a precise prediction; at any practical threshold, most flagged customers won't churn.
Silhouette score is 0.18, so the segments overlap rather than being sharply separated.
Data is a single snapshot. A production model would need time-based validation (train on earlier months, test on later ones) and monitoring for drift.
How to run
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook streamflix_churn_prediction_.ipynb   # prediction + evaluation
python churn_segmentation.py                          # segmentation
The data is loaded directly from the course repository URL in the notebook.

