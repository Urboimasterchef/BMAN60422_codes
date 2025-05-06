import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score, make_scorer, recall_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/yenchengwu/Course Data/Data Analytics/Group Project/cleaned_and_encoded_dataset.csv")

# recall for bad customers (label 1) ─ this is already the default
recall_bad = make_scorer(recall_score, pos_label=1)
# specificity (true‑negative rate for good customers)
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)
specificity_good = make_scorer(specificity, greater_is_better=True)
# train-test-split
X = df.drop(columns=['bad_flag', 'customer_id']).copy()
y = df['bad_flag'].copy()
ids = df['customer_id'].copy()

X_train, X_test, y_train, y_test, id_train, id_test= train_test_split(
    X, y, ids, test_size=0.2, random_state=42, stratify=y
)
sampler = RandomUnderSampler(random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1) # <--- replace this part with other classifier

pipe_rf = Pipeline(steps=[
    ('under', sampler),
    ('clf',   rf)
])

param_grid_rf = {  # <--- edit the grid to have "clf__" before every parameter's name
    'clf__n_estimators': [200, 400],
    'clf__max_depth': [5, 10, 15],
    'clf__min_samples_split': [2, 5, 10],
    'clf__max_features': ['sqrt', 'log2'],
    'clf__min_samples_leaf': [5, 10, 20]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_rf = GridSearchCV(
    estimator = pipe_rf,
    param_grid = param_grid_rf,
    scoring={
        "recall_bad": recall_bad,
        "specificity_good": specificity_good,
    },
    refit="recall_bad",  
    n_jobs=-1,
    cv=cv,
    verbose=2,
)

gs_rf.fit(X_train, y_train)

best_model_rf = gs_rf.best_estimator_
best_params_rf = gs_rf.best_params_
print("Best parameter: ",best_params_rf)
print("Best accuracy train set: ",best_model_rf.score(X_train, y_train))
print("Best accuracy test set: ",best_model_rf.score(X_test, y_test))

train_sizes, train_scores, val_scores = learning_curve(
   best_model_rf, X_train, y_train, cv=cv, scoring=recall_bad, n_jobs=-1)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label="Train")
plt.plot(train_sizes, val_mean, label="Validation")
plt.xlabel("Training Set Size")
plt.ylabel("Recall_Bad")
plt.legend()
plt.title("Learning Curve")
plt.grid(True)
plt.show()

best_y_pred_rf = best_model_rf.predict(X_test)
best_y_prob_rf = best_model_rf.predict_proba(X_test)[:, 1]

# Threshold optimisation
thresholds = np.arange(0.01, 1.0, 0.01)
results_opt = []
for t in thresholds:
    y_pred_thresh = (best_y_prob_rf >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    if recall >= 0.85 and specificity >= 0.70:
        results_opt.append({
            'threshold': t,
            'recall': recall,
            'specificity': specificity,
            'false_negatives': fn,
            'false_positives': fp
        })

opt_df = pd.DataFrame(results_opt)

if not opt_df.empty:
    best_threshold = opt_df.sort_values(by='false_negatives').iloc[0]['threshold']
    print(f"\nBest threshold under constraints: {best_threshold:.2f}")
    print(opt_df.sort_values(by='false_negatives').head())
else:
    # Default to threshold 0.5
    best_threshold = 0.5
    print("\nNo threshold satisfies both recall ≥ 85% and specificity ≥ 70%. Defaulting to threshold = 0.50")

y_pred_thresh = (best_y_prob_rf >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_thresh)
print("\nConfusion matrix (final threshold):\n", cm)
print(classification_report(y_test, y_pred_thresh, digits=3))

print("ROC‑AUC :", roc_auc_score(y_test, best_y_prob_rf).round(3))
print("PR‑AUC  :", average_precision_score(y_test, best_y_prob_rf).round(3))

tn, fp, fn, tp = cm.ravel()
print(f"Recall (bad)         : {tp / (tp + fn):.3f}")
print(f"Specificity (good)   : {tn / (tn + fp):.3f}")
print(f"tn, fp, fn, tp   : {tn:.0f}, {fp:.0f}, {fn:.0f}, {tp:.0f}")


best_imp_rf = best_model_rf.named_steps['clf'].feature_importances_
best_feat_rank_rf = (pd.Series(best_imp_rf, index=X.columns)
             .sort_values(ascending=False)
             .to_frame('importance'))
print(best_feat_rank_rf.head(25))

results = (pd.DataFrame({'customer_id': id_test,
                         'y_true':      y_test,
                         'y_pred':      best_y_pred_rf,
                         'y_prob':      best_y_prob_rf})
           .reset_index(drop=True))

# Logistic Regression
model_logi = LogisticRegression(solver = 'liblinear', max_iter = 100, penalty = 'l1')

pipe_logi = Pipeline(steps=[
    ('under', sampler),
    ('clf',   model_logi)
])

param_grid_logi = [
    # L1 penalty only: needs liblinear or saga
    {
        'clf__penalty': ['l1'],
        'clf__solver': ['liblinear', 'saga'],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__max_iter': [100, 300, 500, 10000],
        'clf__fit_intercept': [True, False],
        'clf__class_weight': [None, 'balanced']
    },
    # L2 penalty: can use all solvers except 'liblinear' with saga's advantage for large data
    {
        'clf__penalty': ['l2'],
        'clf__solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag'],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__max_iter': [100, 300, 500, 10000],
        'clf__fit_intercept': [True, False],
        'clf__class_weight': [None, 'balanced']
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_logi = GridSearchCV(
    estimator = pipe_logi,
    param_grid = param_grid_logi,
    scoring={
        "recall_bad": recall_bad,
        "specificity_good": specificity_good,
    },
    refit="recall_bad",  
    n_jobs=-1,
    cv=cv,
    verbose=2,
)

gs_logi.fit(X_train, y_train)

best_model_logi = gs_logi.best_estimator_
best_params_logi = gs_logi.best_params_
print("Best parameter: ",best_params_logi)
print("Best accuracy train set: ",best_model_logi.score(X_train, y_train))
print("Best accuracy test set: ",best_model_logi.score(X_test, y_test))

train_sizes, train_scores, val_scores = learning_curve(
   best_model_logi, X_train, y_train, cv=cv, scoring=recall_bad, n_jobs=-1)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label="Train")
plt.plot(train_sizes, val_mean, label="Validation")
plt.xlabel("Training Set Size")
plt.ylabel("Recall_Bad")
plt.legend()
plt.title("Learning Curve")
plt.grid(True)
plt.show()

best_y_pred_logi = best_model_logi.predict(X_test)
best_y_prob_logi = best_model_logi.predict_proba(X_test)[:, 1]

# Threshold optimisation
thresholds = np.arange(0.01, 1.0, 0.01)
results_opt = []
for t in thresholds:
    y_pred_thresh = (best_y_prob_logi >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    if recall >= 0.85 and specificity >= 0.70:
        results_opt.append({
            'threshold': t,
            'recall': recall,
            'specificity': specificity,
            'false_negatives': fn,
            'false_positives': fp
        })

opt_df = pd.DataFrame(results_opt)

if not opt_df.empty:
    best_threshold = opt_df.sort_values(by='false_negatives').iloc[0]['threshold']
    print(f"\nBest threshold under constraints: {best_threshold:.2f}")
    print(opt_df.sort_values(by='false_negatives').head())
else:
    # Default to threshold 0.5
    best_threshold = 0.5
    print("\nNo threshold satisfies both recall ≥ 85% and specificity ≥ 70%. Defaulting to threshold = 0.50")
    
y_pred_thresh = (best_y_prob_logi >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_thresh)
print("\nConfusion matrix (final threshold):\n", cm)
print(classification_report(y_test, y_pred_thresh, digits=3))

print("ROC‑AUC :", roc_auc_score(y_test, best_y_prob_logi).round(3))
print("PR‑AUC  :", average_precision_score(y_test, best_y_prob_logi).round(3))

tn, fp, fn, tp = cm.ravel()
print(f"Recall (bad)         : {tp / (tp + fn):.3f}")
print(f"Specificity (good)   : {tn / (tn + fp):.3f}")
print(f"tn, fp, fn, tp   : {tn:.0f}, {fp:.0f}, {fn:.0f}, {tp:.0f}")

results = (pd.DataFrame({'customer_id': id_test,
                         'y_true':      y_test,
                         'y_pred':      best_y_pred_logi,
                         'y_prob':      best_y_prob_logi})
           .reset_index(drop=True))
