Goal: Predict which of 10,000 streaming subscribers would cancel within 30 days (18.9% churn rate) so a retention team could intervene early.

Approach: I engineered 9 behavioral features (e.g., engagement score, payment risk) from 37 raw fields and benchmarked logistic regression, random forest, gradient boosting, and AdaBoost in leak-free scikit-learn pipelines. Because a model that always predicts "no churn" scores 81% accuracy, I selected models on ROC-AUC using stratified 5-fold CV, then tuned gradient boosting with grid search.

Validation: On a held-out stratified test set, the model reached ROC-AUC 0.75. Reviewing my results, I found my reported recall used a weighted average that equaled accuracy; churn-class recall at the default 0.5 threshold was only 8%. I re-evaluated on the churn class and swept thresholds: the top 10% of risk scores churn at 44.5% (2.35× the base rate), and a 0.25 cut-off catches 55% of churners at 38% precision, a trade-off the business can choose.

Extension: K-Means segmentation (label held out) found a payment-failure segment churning at 25%, pointing to targeted interventions.
