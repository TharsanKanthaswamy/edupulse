import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(df):
    """
    Train Random Forest on student data and return model and explanations.
    """
    # Rename column if needed
    if 'dropoutrisk' in df.columns:
        df.rename(columns={'dropoutrisk': 'dropout_risk'}, inplace=True)
    
    X = df.drop(['G1_x','G2_x','G3_x','dropout_risk'], axis=1)
    y = df['dropout_risk']

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    top_features = importances.head(10).index
    feature_reasons = []
    for feature in top_features:
        if feature in df.columns:
            avg_dropout = df.loc[df['dropout_risk'] == 1, feature].mean()
            avg_non_dropout = df.loc[df['dropout_risk'] == 0, feature].mean()
            if avg_dropout < avg_non_dropout:
                reason = (f"{feature}: Dropouts had lower averages ({avg_dropout:.2f}) vs stayers ({avg_non_dropout:.2f}), "
                          "lower values increase dropout risk.")
            else:
                reason = (f"{feature}: Dropouts had higher averages ({avg_dropout:.2f}) vs stayers ({avg_non_dropout:.2f}), "
                          "higher values increase dropout risk.")
        else:
            dropout_rate_with = df.loc[(X[feature] == 1) & (y == 1)].shape[0] / max(1, X[feature].sum())
            dropout_rate_without = df.loc[(X[feature] == 0) & (y == 1)].shape[0] / max(1, (X.shape[0] - X[feature].sum()))
            if dropout_rate_with > dropout_rate_without:
                reason = (f"{feature}: Students with {feature}=1 had higher dropout rate "
                          f"({dropout_rate_with:.1%}) vs others ({dropout_rate_without:.1%}).")
            else:
                reason = (f"{feature}: Students with {feature}=1 had lower dropout rate "
                          f"({dropout_rate_with:.1%}) vs others ({dropout_rate_without:.1%}).")
        feature_reasons.append(reason)

    feature_columns = X_train.columns

    return rf_model, feature_columns, importances, feature_reasons

if __name__ == "__main__":
    df = pd.read_excel("merged_student_data.xlsx")
    
    rf_model, feature_columns, importances, feature_reasons = train_random_forest(df)
    
    print("Top 10 feature importances:")
    print(importances.head(10)*100)
    
    print("\nFeature reasons:")
    for reason in feature_reasons:
        print("-", reason)
    
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(feature_columns, "rf_feature_columns.pkl")
    print("\nModel and features saved successfully!")
