# Credit Card Fraud Detection

### Building a model on an imbalanced dataset

Using a dataset of real, anonymized credit card transactions to simulate the same imbalance banks experience, I built a model to detect fraud.

## Dataset

- Source: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) on Kaggle
- 284,807 transactions made by European cardholders over two days in September 2013
- 492 fraudulent transactions (about 0.17% of the data)
- Features V1 to V28 are PCA-transformed for privacy; `Time` and `Amount` are the original values
- Target: `Class` (1 = fraud, 0 = legitimate)

The dataset is not included in this repo. Download `creditcard.csv` from Kaggle and place it in `[data/]`.

## Approach

1. **Exploratory analysis:** looked at the class distribution and transaction amounts to understand how imbalanced the data is.
2. **Preprocessing:** [scaled `Amount` and `Time`], split the data into training and test sets with stratification so both sets keep the same fraud ratio.
3. **Handling imbalance:** used class weights so the model penalizes missed fraud more heavily, instead of oversampling the minority class.
4. **Model comparison:** trained Logistic Regression, Random Forest, and XGBoost on the same split.
5. **Tuning:** limited Random Forest tree depth to reduce overfitting.

## Evaluation

Accuracy is misleading here. A model that predicts "not fraud" for every transaction would score about 99.8% accuracy while catching zero fraud. Instead, the models are evaluated with:

- **Precision, recall, and F1-score** on the fraud class
- **Precision-recall curve and average precision (PR-AUC)**, which reflect performance on the rare class better than ROC-AUC
- **Confusion matrix** to show missed fraud versus false alarms
- **[Stratified k-fold cross-validation** to check that results are stable across splits]
- **[Threshold tuning** to choose a decision threshold based on validation data instead of the default 0.5]

## Results

| Model | Precision | Recall | F1-score | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | [ ] | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | ~0.82 | [ ] |
| XGBoost | [ ] | [ ] | [ ] | [ ] |

Random Forest performed best overall, reaching an F1-score of about 82% on the fraud class.

[Add PR curve and confusion matrix images here, e.g. `![PR curve](images/pr_curve.png)`]

## Takeaways

Not every dataset can be evaluated the same way. With severe class imbalance, high accuracy means very little, and metrics focused on the minority class are needed to judge whether a model is useful.

## How to Run

```bash
git clone [repo URL]
cd [repo name]
pip install -r requirements.txt
jupyter notebook [notebook name].ipynb
```

## Tools

Python, pandas, scikit-learn, [XGBoost], [matplotlib / seaborn]
