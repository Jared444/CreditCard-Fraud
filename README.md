# Credit Card Fraud Detection

### Building a model on an imbalanced dataset

Using a dataset of real, anonymized credit card transactions to simulate the same imbalance banks experience, I built a model to detect fraud.

## Dataset

- Source: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) on Kaggle
- 284,807 transactions made by European cardholders over two days in September 2013
- 492 fraudulent transactions (about 0.17% of the data)
- Features V1 to V28 are PCA-transformed for privacy; `Time` and `Amount` are the original values
- Target: `Class` (1 = fraud, 0 = legitimate)

The dataset is not included in this repo. Download `creditcard.csv` from Kaggle and place it in the project folder.

## Approach

1. **Exploratory analysis:** checked the class distribution to understand how rare fraud is.
2. **Train/test split:** 80/20 split with stratification so both sets keep the same fraud ratio. The test set was only used once, for the final evaluation.
3. **Handling imbalance:** used class weights (`class_weight="balanced"` for Random Forest, `scale_pos_weight` for XGBoost) so the models penalize missed fraud more heavily, instead of oversampling the minority class.
4. **Model comparison:** compared Random Forest and XGBoost using 5-fold stratified cross-validation on the training set.
5. **Threshold tuning:** collected out-of-fold predicted probabilities on the training set and chose the decision threshold that maximized F1, instead of using the default 0.5.
6. **Final evaluation:** retrained the selected model on the full training set and evaluated it once on the held-out test set.

## Evaluation

Accuracy is misleading here. A model that predicts "not fraud" for every transaction would score about 99.8% accuracy while catching zero fraud. Instead, the models are evaluated with:

- **PR-AUC (average precision)** to compare models across all thresholds, which reflects performance on the rare class better than ROC-AUC
- **Precision, recall, and F1-score** on the fraud class at the tuned threshold
- **Confusion matrix** to show missed fraud versus false alarms
- **Stratified 5-fold cross-validation** to check that results are stable across splits

## Results

### Cross-validation (training set, PR-AUC)

| Model | Mean PR-AUC | Std |
|---|---|---|
| Random Forest | 0.832 | 0.022 |
| XGBoost | 0.855 | 0.022 |

Using the same folds for both models, XGBoost scored higher in all 5 folds, so it was selected as the final model.

### Final test set (XGBoost)

| Threshold | Precision | Recall | F1-score | PR-AUC |
|---|---|---|---|---|
| 0.925 | 0.941 | 0.816 | 0.874 | 0.887 |

The model caught 80 of 98 fraudulent transactions with 5 false alarms out of 56,864 legitimate ones. The test F1 (0.874) was close to the cross-validated F1 (0.862), which suggests the tuned threshold generalizes to unseen data.

## Takeaways

Not every dataset can be evaluated the same way. With severe class imbalance, high accuracy means very little, and metrics focused on the minority class are needed to judge whether a model is useful.

Class weighting also changes what the predicted probabilities mean. With `scale_pos_weight`, XGBoost's fraud probabilities were pushed upward, so the best threshold was 0.925 rather than the default 0.5. Choosing the threshold from out-of-fold training predictions, instead of guessing or tuning on the test set, gave a better and more trustworthy result.

## How to Run

```bash
git clone [repo URL]
cd CreditCard-Fraud
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook fraud_detector.ipynb
```

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost