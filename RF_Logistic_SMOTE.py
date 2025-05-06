import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score, make_scorer, recall_score)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

X_test = np.array(pd.read_csv("/Users/yenchengwu/Course Data/Data Analytics/Group Project/SMOTE dataset/x_test_smote2.csv"))
y_test = pd.read_csv("/Users/yenchengwu/Course Data/Data Analytics/Group Project/SMOTE dataset/y_test_smote2.csv").values.ravel()
X_train = np.array(pd.read_csv("/Users/yenchengwu/Course Data/Data Analytics/Group Project/SMOTE dataset/x_train_smote2.csv"))
y_train = pd.read_csv("/Users/yenchengwu/Course Data/Data Analytics/Group Project/SMOTE dataset/y_train_smote2.csv").values.ravel()


# recall for bad customers (label 1) ─ this is already the default
recall_bad = make_scorer(recall_score, pos_label=1)

# specificity (true‑negative rate for good customers)
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)
specificity_good = make_scorer(specificity, greater_is_better=True)

# Random Forest
rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)

pipe_rf = Pipeline(steps=[
    ('clf',   rf)
])

pipe_rf.fit(X_train, y_train)

y_pred_rf = pipe_rf.predict(X_test)
y_prob_rf = pipe_rf.predict_proba(X_test)[:, 1]

rf_params = {k: v for k, v in pipe_rf.get_params().items()
             if k.startswith('clf__') and
                any(p in k for p in ['n_estimators', 'max_depth', 'min_samples_split'])}
print("Random Forest Parameters:", rf_params)

print(classification_report(y_test, y_pred_rf, digits=3))
print("ROC‑AUC :", roc_auc_score(y_test, y_prob_rf).round(3))
print("PR‑AUC  :", average_precision_score(y_test, y_prob_rf).round(3))

cm_rf = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = cm_rf.ravel()
print(f"Recall (bad)         : {tp / (tp + fn):.3f}")
print(f"Specificity (good)   : {tn / (tn + fp):.3f}")
print(f"tn, fp, fn, tp   : {tn:.0f}, {fp:.0f}, {fn:.0f}, {tp:.0f}")

results = (pd.DataFrame({'y_true':      y_test,
                         'y_pred':      y_pred_rf,
                         'y_prob':      y_prob_rf})
           .reset_index(drop=True))

param_grid_rf = {
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

results = (pd.DataFrame({'y_true':      y_test,
                         'y_pred':      best_y_pred_rf,
                         'y_prob':      best_y_prob_rf})
           .reset_index(drop=True))

# Logistic Regression
model_logi = LogisticRegression(solver = 'liblinear', max_iter = 100, penalty = 'l1')

pipe_logi = Pipeline(steps=[
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

results = (pd.DataFrame({'y_true':      y_test,
                         'y_pred':      best_y_pred_logi,
                         'y_prob':      best_y_prob_logi})
           .reset_index(drop=True))

