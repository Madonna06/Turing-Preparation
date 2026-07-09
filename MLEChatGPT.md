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

Can refer : https://medium.com/@bhatadithya54764118/day-30-encoding-categorical-variables-one-hot-label-and-target-encoding-23c2e38eeb6a

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

How it works: It looks at the feature space of the minority class. For a given fraud data point, it finds its nearest neighbors (using k-Nearest Neighbors), draws a line between them in the multi-dimensional space, and places a new synthetic data point somewhere along that line.

Pros: Prevents overfitting by introducing new, unobserved feature combinations to the model.

Cons: Can cause severe overgeneralization and noise. If a fraud data point is an outlier sitting right next to legitimate transactions, SMOTE will draw lines between them, creating synthetic fraud points inside the legitimate data zone, blurring the decision boundary.

Random oversampling

How it works: It randomly selects existing minority class examples (fraud) and duplicates them exactly until the desired class ratio is achieved.

Pros: Incredibly simple to implement and requires zero mathematical configuration.

Cons: Leads to severe overfitting. The model essentially memorizes the exact copied data points rather than learning the underlying patterns of fraud.

Random undersampling

How it works: It randomly selects and deletes examples from the majority class (legitimate) until the dataset matches the size of the minority class.

Pros: Radically reduces training time and data storage requirements because the overall dataset becomes much smaller.

Cons: Causes massive information loss. You throw away valuable data regarding what constitutes a "normal" transaction, which can cause the model to flag a high number of false positives in production.
Class weights

How it works: During the training optimization phase (like Gradient Descent), the loss calculated for a misclassified minority sample is multiplied by a heavy weight factor. If your ratio is 1:100, a missed fraud case costs the model 100 times more than a missed clean transaction.

Pros: Highly effective for linear algorithms, logistic regression, and standard neural networks without changing the underlying dataset.

Cons: Can cause the model to become overly cautious, severely hurting Precision (generating too many false alerts).
Focal Loss

Focal Loss is a dynamic loss function originally designed for computer vision but highly adapted for tabular fraud models.

How it works: Standard loss functions treat all errors relatively equally. Focal Loss adds a modulating factor \((1 - p)^\gamma\) to the down-weight easy-to-classify examples. If the model is 99% confident a transaction is legitimate, Focal Loss drops the importance of that row to near zero. Instead, it forces the model's entire attention onto hard, ambiguous examples (the gray area where fraud hides).

Pros: Stops the massive volume of easy legitimate transactions from overwhelming the model's gradients during training.

Cons: Requires fine-tuning an extra hyperparameter (\(\gamma \), gamma), and is computationally heavier to calculate.

Balanced Random Forest

A specialized variation of the standard Random Forest algorithm.

How it works: When building each individual tree in the forest, the algorithm takes a bootstrap sample of the data. However, instead of taking a random sample of the whole dataset, it performs Random Undersampling independently for each tree. Each tree trains on a balanced subset containing all minority samples and a matching, randomly selected chunk of majority samples.

Pros: Retains the high predictive power of Random Forest while bypassing the information-loss penalty of standard global undersampling, because every piece of majority data is eventually seen by different trees.

Cons: Can be slow to train and has a larger memory footprint than standard tree models.

XGBoost scale_pos_weight

This is a built-in hyperparameter specifically inside the XGBoost library designed to manage extreme imbalance.

How it works: It controls the balance of positive (fraud) and negative (legitimate) weights. It directly scales the gradients of the positive class. The standard formula to calculate it is:\(\text{scale\_pos\_weight}=\frac{\text{Total\ Number\ of\ Negative\ (Legitimate)\ Samples}}{\text{Total\ Number\ of\ Positive\ (Fraud)\ Samples}}\)If you have 99,000 legitimate rows and 1,000 fraud rows, you set scale_pos_weight = 99.

Pros: Simple, clean hyperparameter implementation that scales directly with the data ratio. It drastically improves Recall for tree boosting.

Cons: It completely distorts the raw predicted probabilities. Instead of outputting a true probability (e.g., 0.05 risk), the model's raw score outputs will skew incredibly high. If you need calibrated probabilities for business logic risk thresholds, you must manually post-calibrate the model outputs.

How scale_pos_weight Changes the Math

When you set scale_pos_weight = 99, XGBoost manually rewrites its objective loss function. It takes every single gradient (\(g\)) and Hessian (\(h\)) belonging to a positive sample (Fraud) and multiplies it directly by 99.

Majority Row Error: Contributes \(1 \times g\) to the tree split calculation.

Minority Row Error: Contributes \(99 \times g\) to the tree split calculation.

Now, let's look at the balance of power inside the leaf node calculation:

Total power of Majority Class: \(99,000 \times 1 = 99,000\)

Total power of Minority Class: \(1,000 \times 99 = 99,000\)

Mathematically, the two classes now possess equal weight. Even though there are physically 99 times more majority rows in your RAM, the optimization algorithm treats a single error on a fraud case as 99 times more devastating than an error on a legitimate transaction.

How SMOTE Actually Works (Deterministic Lines)

SMOTE looks at a real fraud point.It finds its 5 closest neighbors using spatial distance (Euclidean distance).It randomly picks one of those neighbors and draws a strict straight line between them.It places a new synthetic point randomly somewhere along that line.Because it only connects existing data points with straight lines, SMOTE creates data in rigid, geometric paths. It has no concept of probability distributions.

SMOTE uses k-Nearest Neighbors (k-NN) because it is a simple, effective way to find data points that belong to the same localized group or pattern.

Q7.

When would you NOT use SMOTE?

Expected:

High-dimensional data

High-dimensional data contains a massive number of columns or features relative to the number of rows (e.g., genetic markers, image pixels, or hundreds of engineered features).

Why SMOTE Fails: SMOTE relies entirely on k-Nearest Neighbors (k-NN) to select the data points it connects to create synthetic samples. In high-dimensional spaces, a phenomenon known as the "Curse of Dimensionality" occurs: the mathematical distance (Euclidean distance) between all data points converges and becomes almost exactly equal.

The Consequence: Because every point appears roughly equidistant from every other point, the k-NN algorithm can no longer reliably identify true geometric neighbors. SMOTE ends up picking random, unrelated minority samples to draw lines between, introducing massive structural noise into your model.

Sparse features

Sparse datasets are those where the vast majority of cells in your data matrix contain zeros (e.g., bag-of-words text representations, TF-IDF vectors, or user-item interaction matrices in recommendation engines).

Why SMOTE Fails: SMOTE creates synthetic points by picking a point along a straight line connecting a sample to its neighbor: Synthetic = Point1 + random_ratio * (Point2 - Point1).

The Consequence: If Point1 has a zero in a column and Point2 has a non-zero value, the subtraction and multiplication math will result in a continuous decimal value (like 0.342) for that feature in the new synthetic row. This instantly destroys the sparsity structure of your data. A matrix that was 99% zeros becomes filled with synthetic fractional noise, causing massive memory bloat and breaking algorithms designed strictly for sparse data (like Naive Bayes or linear models).

Naive Bayes breaks because its probability formulas require discrete frequencies, which are corrupted by SMOTE's fractional interpolations.

Linear Models break because they lose their sparse matrix formatting optimizations, causing memory crashes and computational exhaustion.


Categorical variables

Categorical data consists of discrete values or text strings, such as payment_method (Credit, Debit) or device_id.

Why SMOTE Fails: 

Standard SMOTE is built purely for continuous numerical spaces where drawing a line between two numbers makes mathematical sense. It does not know how to handle distinct categories.

The Consequence: If SMOTE attempts to interpolate between payment_method = 1 (Credit) and payment_method = 3 (PayPal), the math will generate a fractional category like 1.72. A decision tree cannot interpret what a "1.72" payment method means.

Note: While variants like SMOTE-NC (Nominal Continuous) exist to handle mixed data, they rely on calculating heavy statistical mode values across categorical distributions, which drastically slows down computation and still struggles with high-cardinality columns like customer_id.

Huge datasets

Huge datasets contain millions of rows of transaction, user, or sensor data.

Why SMOTE Fails: SMOTE has a high computational complexity because it must calculate a full k-NN distance matrix for the minority class to find nearest neighbors before it can even begin generating new rows.

The Consequence: Running a k-NN search across millions of records is incredibly expensive in terms of both CPU processing time and RAM usage. Trying to scale SMOTE to massive enterprise datasets can bottleneck your machine learning pipeline, cause out-of-memory (OOM) system crashes, and waste hours of training time for negligible performance gains.

Risk of creating unrealistic samples

This is the most dangerous conceptual flaw of SMOTE, especially in strict fields like fraud detection, healthcare, or financial compliance.

Why SMOTE Fails: SMOTE assumes that the space between two real minority points is safe, valid territory to place a new point. However, real-world data distributions are often highly non-linear, multi-modal, or fragmented into small, isolated clusters.

The Consequence: If a dataset has a strict physical or logical boundary (for example, a hard business rule where transaction_amount cannot exceed a certain threshold for a specific account type), SMOTE ignores this rule. It draws lines blindly across empty space, frequently creating impossible, physically unrealistic samples (e.g., a synthetic fraud user who transacts 500 times in 1 second, or a medical patient with an impossible combination of vital signs). This teaches your machine learning model to look for patterns that can never actually happen in the real world, increasing false positives in production.

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

The F1-score completely ignores True Negatives (TN). It only cares about how well you handle the positive class alarms. Because the model completely failed to identify a single clean transaction, its baseline architectural accuracy collapsed to 50%, while its F1-score remained mathematically sky-high.

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

In a highly imbalanced fraud dataset, the number of Total Actual Negatives is overwhelmingly massive compared to the positive class. Because this denominator is so vast, even a huge number of False Positives will fail to move the fraction

Model may appear good despite poor fraud detection.

Q20.

Which is better for fraud detection?

ROC-AUC or PR-AUC?

Expected:

PR-AUC.

Because it focuses on minority class performance.

If you evaluate this exact same model using Precision (which ignores the massive True Negatives column and focuses strictly on the accuracy of your active alarms), the operational failure becomes instantly clea

Q21.

Your model has

ROC-AUC = 0.98

Precision = 0.07

Would you deploy it?

Expected:

No.

Investigators would receive too many false alarms.

The model is generating an overwhelming avalanche of false alarms that would break your operations team.

A Precision of 0.07 (7%) means that out of every 100 fraud alerts the model triggers, only 7 are actual fraud. The other 93 alerts are completely false alarms—innocent customers who are getting their credit cards blocked or accounts frozen unnecessarily.

The high ROC-AUC of 0.98 makes the model look nearly perfect on paper. However, as we discussed, ROC-AUC is completely blinded by the massive number of legitimate transactions in an imbalanced dataset.If your company processes 1,000,000 transactions a day, a 1% False Positive Rate sounds small to ROC-AUC, but it translates to 10,000 false alarms every single day.

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
