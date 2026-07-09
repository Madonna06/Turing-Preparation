Feature Engineering
Q1. You have a fraud dataset with these columns:
transaction_amount
timestamp
merchant_id
customer_id
device_id
location
payment_method

What features would you create?

Expected discussion

Time since previous transaction
Number of transactions in last hour/day
Average spending
Merchant frequency
Device sharing
Velocity features
Geographic distance from previous transaction
Hour of day/weekend/night
Customer lifetime statistics
Q2.

You notice merchant_id has 500,000 unique values.

How would you encode it?

Expected answers

Target encoding (carefully)
Frequency encoding
Embeddings
Hash encoding
Avoid naive one-hot encoding

Follow-up:

Why is target encoding dangerous?

Expected:
Because it causes target leakage if computed on the full dataset. Use K-fold target encoding.

Q3.

How do you engineer features from timestamps?

Possible answers:

Hour
Day of week
Weekend
Holiday
Time since previous purchase
Time between transactions
Cyclic encoding using sin/cos
Q4.

How do you know whether a new engineered feature is useful?

Expected:

Feature importance
Mutual information
SHAP values
Cross-validation improvement
Correlation analysis
Handling Imbalanced Data

Suppose:

Fraud = 0.2%

Legitimate = 99.8%
Q5.

Would you use accuracy?

Why or why not?

Expected:

No.

A model predicting "Not Fraud" every time achieves 99.8% accuracy.

Accuracy is misleading.

Q6.

How would you handle this imbalance?

Expected answers:

SMOTE
Random oversampling
Random undersampling
Class weights
Focal Loss
Balanced Random Forest
XGBoost scale_pos_weight
Q7.

When would you NOT use SMOTE?

Expected:

High-dimensional data
Sparse features
Categorical variables
Huge datasets
Risk of creating unrealistic samples
Q8.

If fraud is only 0.1%, would you oversample or use class weights?

Expected discussion:

Depends.

Usually:

Start with class weights
Try XGBoost LightGBM CatBoost
Compare with oversampling
Q9.

How do class weights work?

Expected:

Increase penalty for misclassifying minority class.

Loss becomes

Weighted Loss

Fraud mistakes cost more.
Evaluation Metrics
Q10.

What is Precision?

Expected:

TP / (TP + FP)

Meaning:

Among predicted frauds,
how many are actually fraud?

Q11.

What is Recall?

Expected:

TP / (TP + FN)

Meaning:

Among actual frauds,
how many did we catch?

Q12.

Fraud detection:

Would you prioritize Precision or Recall?

Expected discussion:

Usually Recall.

Missing fraud is expensive.

But if Precision is extremely low,
investigators waste time.

Need balance.

Q13.

When is Precision more important?

Expected examples:

Spam filtering
Medical diagnosis with expensive treatment
Sending police alerts
Human review systems
Q14.

When is Recall more important?

Expected:

Fraud detection
Cancer detection
Terrorist detection
Disease screening
Q15.

Explain F1-score.

Expected:

Harmonic mean of Precision and Recall.

2PR / (P + R)

Useful when both matter.

Q16.

Can F1-score be high while accuracy is low?

Yes.

Explain with an example.

Q17.

Can accuracy be 99% while Recall is 0?

Yes.

Fraud is very rare.

Model predicts all transactions as legitimate.

ROC-AUC
Q18.

What does ROC-AUC measure?

Expected:

Ability to distinguish positive and negative classes across thresholds.

Q19.

Why can ROC-AUC be misleading on imbalanced datasets?

Expected:

False Positive Rate remains small because there are many negatives.

Model may appear good despite poor fraud detection.

Q20.

Which is better for fraud detection?

ROC-AUC or PR-AUC?

Expected:

PR-AUC.

Because it focuses on minority class performance.

Q21.

Your model has

ROC-AUC = 0.98

Precision = 0.07

Would you deploy it?

Expected:

No.

Investigators would receive too many false alarms.

Threshold Questions
Q22.

Why isn't the threshold always 0.5?

Expected:

Threshold depends on business cost.

For fraud:

Lower threshold

Higher Recall

Lower Precision

Q23.

How would you choose the threshold?

Expected:

Precision-Recall curve
Business cost matrix
F1 optimization
Expected financial loss
Scenario Questions
Q24.

Model A

Precision = 95%

Recall = 20%

Model B

Precision = 60%

Recall = 85%

Which would you choose?

Expected discussion:

Depends on business.

Fraud teams usually prefer Model B.

Q25.

False negatives cost $10,000.

False positives cost $10.

How would that affect your model?

Expected:

Increase Recall.

Lower threshold.

Accept more false positives.

Q26.

Your fraud model performs well offline but poorly in production.

Possible reasons?

Expected:

Data drift
Concept drift
Target leakage
Different customer behavior
Missing production features
Wrong preprocessing
Label delay
Q27.

How would you detect concept drift?

Expected:

Monitor Precision/Recall
PSI
KL divergence
Feature distributions
Drift detection libraries
Q28.

What would you monitor after deployment?

Expected:

Precision
Recall
Fraud rate
Feature drift
Prediction distribution
Latency
False positives
Model confidence
Coding-Oriented Questions
Q29.

How would you compute Precision in Python without using scikit-learn?

tp = sum((y_true == 1) & (y_pred == 1))
fp = sum((y_true == 0) & (y_pred == 1))

precision = tp / (tp + fp)
Q30.

How would you split fraud data into train and test?

Expected:

Use stratified splitting to preserve the fraud/non-fraud ratio.

Rapid-Fire Questions (Common at Turing)
Why is one-hot encoding bad for high-cardinality features?
What is target leakage?
Difference between feature selection and feature extraction?
What is data drift vs. concept drift?
When would you use SMOTE?
Why is PR-AUC often preferred over ROC-AUC for fraud detection?
How do class weights influence training?
When should you optimize for Precision instead of Recall?
Why use Stratified K-Fold instead of standard K-Fold for imbalanced data?
How would you explain Precision and Recall to a non-technical stakeholder?

These questions reflect the style often used in ML interviews: practical, scenario-driven, and focused on explaining trade-offs rather than reciting definitions.
