import pandas as pd
from sklearn.model_selection import train_test_split

def clean_and_preprocess(df):
    """
    Cleans and preprocesses merged student data.
    
    Args:
        df (pd.DataFrame): Merged student dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe with feature engineering and target variable
    """
    df = df.drop_duplicates()  # Remove duplicates if any
    
    # Fill missing numeric columns
    numeric_cols = ['G1_x', 'G2_x', 'G3_x', 'absences_x', 'studytime_x', 'failures_x', 'famrel_x', 'freetime_x', 'goout_x', 'Dalc_x', 'Walc_x', 'health_x']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill binary columns and map to 0/1
    binary_cols = ['schoolsup_x', 'famsup_x', 'paid_x', 'activities_x', 'nursery', 'higher_x', 'internet', 'romantic_x']
    for col in binary_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].map({'yes':1, 'no':0})
    
    # One-hot encode nominal features
    df = pd.get_dummies(df, columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian_x'])
    
    # Feature engineering
    df['avg_grade'] = (df['G1_x'] + df['G2_x']) / 2
    df['failure_flag'] = (df['failures_x'] > 0).astype(int)
    df['attendance_ratio'] = 1 - (df['absences_x'] / 50)
    df['attendance_ratio'] = df['attendance_ratio'].clip(0,1)
    
    # Target variable for dropout risk
    df['dropout_risk'] = ((df['G3_x'] < 10) | (df['absences_x'] > 20)).astype(int)
    
    return df

def split_data(df):
    """
    Splits cleaned data into train/test sets.
    
    Args:
        df (pd.DataFrame): Cleaned student data
    
    Returns:
        X_train, X_test, y_train, y_test: splits
    """
    X = df.drop(['G1_x', 'G2_x', 'G3_x', 'dropout_risk'], axis=1)
    y = df['dropout_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
