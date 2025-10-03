import pandas as pd
import numpy as np
import random

# --- Configuration ---
NUM_STUDENTS = 180
OUTPUT_FILENAME = 'demo_student_data_encoded.csv'

# Define the risk profile distribution
risk_profiles = (
    ['Low-Risk'] * int(NUM_STUDENTS * 0.25) +
    ['Moderate-Risk'] * int(NUM_STUDENTS * 0.45) +
    ['High-Risk'] * int(NUM_STUDENTS * 0.30)
)
while len(risk_profiles) < NUM_STUDENTS:
    risk_profiles.append('Moderate-Risk')
random.shuffle(risk_profiles)


# --- Risk Calculation Function ---
def calculate_dropout_risk(student):
    """Calculates a dropout risk score (0-100) based on student attributes."""
    if student['profile'] == 'High-Risk':
        risk = 65
    elif student['profile'] == 'Moderate-Risk':
        risk = 35
    else: # Low-Risk
        risk = 10

    # Adjust risk based on key factors
    risk += (12 - student['G3']) * 3
    risk += student['failures'] * 12
    risk -= (student['studytime'] - 2) * 5
    risk += student['absences'] * 0.75
    if student['higher'] == 'no':
        risk += 15
    if student['famsup'] == 'no':
        risk += 5
    risk += (3 - student['famrel']) * 2
    risk += (student['Walc'] - 2) * 1.5
    if student['romantic'] == 'yes':
        risk += 3
    risk += np.random.normal(0, 3)
    return int(np.clip(risk, 5, 98))


# --- Data Generation Function ---
def generate_student_record(profile):
    record = {'profile': profile}
    
    # Profile-based data
    if profile == 'High-Risk':
        record['failures'] = np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])
        record['G1'] = np.random.randint(0, 11)
        record['absences'] = np.random.randint(15, 31)
        record['studytime'] = np.random.choice([1, 2], p=[0.7, 0.3])
        record['famrel'] = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        record['higher'] = np.random.choice(['no', 'yes'], p=[0.6, 0.4])
        record['Walc'] = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
    elif profile == 'Moderate-Risk':
        record['failures'] = np.random.choice([0, 1], p=[0.8, 0.2])
        record['G1'] = np.random.randint(10, 15)
        record['absences'] = np.random.randint(5, 16)
        record['studytime'] = np.random.choice([2, 3], p=[0.6, 0.4])
        record['famrel'] = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
        record['higher'] = np.random.choice(['yes', 'no'], p=[0.85, 0.15])
        record['Walc'] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
    else: # Low-Risk
        record['failures'] = 0
        record['G1'] = np.random.randint(14, 21)
        record['absences'] = np.random.randint(0, 6)
        record['studytime'] = np.random.choice([3, 4], p=[0.5, 0.5])
        record['famrel'] = np.random.choice([4, 5], p=[0.4, 0.6])
        record['higher'] = 'yes'
        record['Walc'] = np.random.choice([1, 2], p=[0.7, 0.3])

    # Grade calculation
    g2_change = (record['studytime'] * 0.5) - (record['failures'] * 1.5) - (record['absences'] * 0.1)
    record['G2'] = int(np.clip(round(record['G1'] + g2_change), 0, 20))
    g3_change = (record['studytime'] * 0.6) - (record['failures'] * 1.8) - (record['absences'] * 0.15)
    record['G3'] = int(np.clip(round(((record['G1'] * 0.3) + (record['G2'] * 0.7)) + g3_change), 0, 20))
    
    # Standard demographic and other data
    record['school'] = np.random.choice(['GP', 'MS'], p=[0.7, 0.3])
    record['sex'] = np.random.choice(['M', 'F'])
    record['age'] = int(np.clip(np.random.normal(17.5, 1.2), 15, 22))
    record['address'] = np.random.choice(['U', 'R'], p=[0.75, 0.25])
    record['famsize'] = np.random.choice(['LE3', 'GT3'])
    record['Pstatus'] = np.random.choice(['T', 'A'], p=[0.85, 0.15])
    record['Medu'] = np.random.choice([0, 1, 2, 3, 4])
    record['Fedu'] = np.random.choice([0, 1, 2, 3, 4])
    record['Mjob'] = np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'])
    record['Fjob'] = np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'])
    record['reason'] = np.random.choice(['home', 'reputation', 'course', 'other'])
    record['guardian'] = np.random.choice(['mother', 'father', 'other'])
    record['traveltime'] = np.random.choice([1, 2, 3, 4])
    record['schoolsup'] = np.random.choice(['yes', 'no'])
    record['famsup'] = np.random.choice(['yes', 'no'], p=[0.7, 0.3])
    record['paid'] = np.random.choice(['yes', 'no'], p=[0.3, 0.7])
    record['activities'] = np.random.choice(['yes', 'no'])
    record['nursery'] = np.random.choice(['yes', 'no'], p=[0.8, 0.2])
    record['internet'] = np.random.choice(['yes', 'no'], p=[0.85, 0.15])
    record['romantic'] = np.random.choice(['yes', 'no'], p=[0.4, 0.6])
    record['freetime'] = np.random.randint(1, 6)
    record['goout'] = np.random.randint(1, 6)
    record['Dalc'] = np.random.randint(1, 6)
    record['health'] = np.random.randint(1, 6)
    
    # Calculate dropout risk
    record['dropout_risk'] = calculate_dropout_risk(record)
    return record

# --- Main Execution ---
if __name__ == "__main__":
    student_data = [generate_student_record(profile) for profile in risk_profiles]
    df = pd.DataFrame(student_data)
    
    # Set the exact column order before encoding
    column_order = [
        'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
        'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
        'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
        'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
        'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'dropout_risk'
    ]
    df = df[column_order]

    # --- ONE-HOT ENCODING ---
    # Identify categorical columns to be encoded
    categorical_cols = [
        'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
        'nursery', 'higher', 'internet', 'romantic'
    ]
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save the encoded dataframe to CSV
    df_encoded.to_csv(OUTPUT_FILENAME, index=False, sep=';')

    print(f"✅ Successfully generated one-hot encoded file: '{OUTPUT_FILENAME}'")
    print("\nData Head (Encoded):")
    # Display the first 5 rows of the new encoded dataframe
    print(df_encoded.head())
    print(f"\nOriginal number of columns: {len(df.columns)}")
    print(f"Encoded number of columns: {len(df_encoded.columns)}")