import pandas as pd

def merge_students(df1, df2):
    """
    Merges two student DataFrames on the defined common columns.
    
    Args:
        df1 (pd.DataFrame): First student dataframe (e.g., student-mat)
        df2 (pd.DataFrame): Second student dataframe (e.g., student-por)
    
    Returns:
        pd.DataFrame: Merged dataframe containing students with combined info
    """
    merge_cols = [
        "school","sex","age","address","famsize","Pstatus",
        "Medu","Fedu","Mjob","Fjob","reason","nursery","internet"
    ]
    merged_df = pd.merge(df1, df2, on=merge_cols)
    print(f"Merged data length: {len(merged_df)}")
    return merged_df
