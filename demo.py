"""
تمرین حرفه‌ای هایپرپارامترها در Machine Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 شروع پروژه هایپرپارامترها...\n")

# 1. لود داده Iris
print("📊 لود داده Iris...")
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. Grid Search برای SVM
print("\n🔍 Grid Search برای SVM...")
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(
    SVC(random_state=42), param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1
)
svm_grid.fit(X_train, y_train)
print(f"بهترین هایپرپارامتر SVM: {svm_grid.best_params_}")
print(f"بهترین امتیاز: {svm_grid.best_score_:.3f}")

# 3. Random Search برای RandomForest
print("\n🎲 Random Search برای RandomForest...")
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), 
    param_dist_rf, n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
rf_random.fit(X_train, y_train)
print(f"بهترین هایپرپارامتر RF: {rf_random.best_params_}")
print(f"بهترین امتیاز: {rf_random.best_score_:.3f}")

# 4. تست نهایی
print("\n✅ تست نهایی...")
svm_best = svm_grid.best_estimator_
rf_best = rf_random.best_estimator_

svm_acc = accuracy_score(y_test, svm_best.predict(X_test))
rf_acc = accuracy_score(y_test, rf_best.predict(X_test))

print(f"SVM دقت تست: {svm_acc:.3f}")
print(f"RF دقت تست: {rf_acc:.3f}")

# 5. ذخیره مدل‌ها
joblib.dump(svm_best, 'best_svm_model.pkl')
joblib.dump(rf_best, 'best_rf_model.pkl')
print("\n💾 مدل‌ها ذخیره شدند!")

# 6. مقایسه بصری
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(['SVM Grid', 'RF Random'], [svm_grid.best_score_, rf_random.best_score_])
ax1.set_title('بهترین امتیاز Validation')
ax1.set_ylabel('Accuracy')

ax2.bar(['SVM Test', 'RF Test'], [svm_acc, rf_acc])
ax2.set_title('دقت روی Test Set')
ax2.set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig('hyperparams_comparison.png')
plt.show()

print("\n🎉 پروژه کامل شد! نتایج در hyperparams_comparison.png")