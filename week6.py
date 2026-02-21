import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load train and test data (from your folder)
# --------------------------------------------------

train_df = pd.read_csv("train_features.csv")
test_df  = pd.read_csv("test_features.csv")

print(train_df.shape, test_df.shape)

# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

target = "runs"

X_train = train_df.drop(columns=[target])
X_test  = test_df.drop(columns=[target])

X_train = X_train.select_dtypes(include=["int64","float64"])
X_test  = X_test.select_dtypes(include=["int64","float64"])

lgb_model = joblib.load("lgb_model.pkl")

X_sample = X_test.sample(
    n=min(300, len(X_test)),
    random_state=42
)

explainer = shap.TreeExplainer(
    lgb_model,
    feature_perturbation="tree_path_dependent"
)

shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, show=False)

plt.tight_layout()
plt.savefig("shap_summary_lgbm.png", dpi=300)
plt.close()

print("Saved shap_summary_lgbm.png")

# --------------------------------------------------
# 7. Feature importance from model
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": lgb_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature importance:")
print(feature_importance)

feature_importance.to_csv("feature_importance_lgbm.csv", index=False)

print("\nSaved feature_importance_lgbm.csv")
