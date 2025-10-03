import pandas as pd
import numpy as np
import shap
import joblib

class SHAPAnalyzer:
    def __init__(self, model_path, data_df):
        self.model = joblib.load(model_path)
        self.data = data_df.copy()
        
        # Rename column if needed
        if 'dropoutrisk' in self.data.columns:
            self.data.rename(columns={'dropoutrisk': 'dropout_risk'}, inplace=True)
        
        print("SHAP Data columns:", self.data.columns.tolist())
        
        if 'student_id' not in self.data.columns:
            self.data['student_id'] = range(len(self.data))

        self.X = self.data.drop(['G1_x', 'G2_x', 'G3_x', 'dropout_risk'], axis=1)
        self.X_encoded = pd.get_dummies(self.X, drop_first=True)
        
        # Load feature columns from the saved model
        try:
            feature_columns = joblib.load("rf_feature_columns.pkl")
            self.X_encoded = self.X_encoded.reindex(columns=feature_columns, fill_value=0)
        except:
            print("Warning: Could not load feature columns file")
        
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.X_encoded, check_additivity=False)
        self.shap_vals_pos = self.shap_values.values[:, :, 1]
        self.expected_value = self.explainer.expected_value[1]
    
    def explain_student(self, student_id):
        if student_id not in self.data['student_id'].values:
            return {"error": f"Student ID {student_id} not found."}
        
        idx = self.data.index[self.data['student_id'] == student_id][0]
        student_shap = self.shap_vals_pos[idx]
        student_features = self.X_encoded.iloc[idx]
        
        total_risk = self.expected_value + student_shap.sum()
        total_risk = np.clip(total_risk, 0, 1)
        
        explanations = []
        for feat, val, shap_val in zip(student_features.index, student_features.values, student_shap):
            if abs(shap_val) < 0.001:  # Skip very small contributions
                continue
            explanations.append({
                "feature": feat,
                "feature_value": val,
                "shap_value": float(shap_val),
                "interpretation": "increases dropout risk" if shap_val > 0 else "decreases dropout risk"
            })
        
        # Sort by absolute SHAP value
        explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            "student_id": int(student_id),
            "predicted_dropout_risk_percent": round(total_risk * 100, 2),
            "feature_explanations": explanations[:10]  # Top 10
        }

# Example usage
if __name__ == "__main__":
    data = pd.read_excel("merged_student_data.xlsx")
    shap_analyzer = SHAPAnalyzer("rf_model.pkl", data)
    student_id = 0  # Test with first student
    explanation = shap_analyzer.explain_student(student_id)
    if "error" in explanation:
        print(explanation["error"])
    else:
        print(f"Dropout risk for Student {student_id}: {explanation['predicted_dropout_risk_percent']}%")
        for feat_exp in explanation["feature_explanations"]:
            print(f"- {feat_exp['feature']}={feat_exp['feature_value']} {feat_exp['interpretation']} (SHAP={feat_exp['shap_value']:.4f})")
