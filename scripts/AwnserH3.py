import pandas as pd
import glob
import re
from scipy.stats import ttest_ind

# Load datasets
csedm_df = pd.read_csv("CSV_files/CodeStates.csv")
llm_files = glob.glob("prompting/*/processed_responses.csv")
llm_df = pd.concat([pd.read_csv(f) for f in llm_files], ignore_index=True)

# Choose code column
def extract_code_column(df):
    for col in ["Code", "code", "Extracted_Code"]:
        if col in df.columns:
            return df[col].dropna()
    return pd.Series(dtype=str)

csedm_code = extract_code_column(csedm_df)
llm_code = extract_code_column(llm_df)

# Helper to count pattern occurrences
def count_occurrences(code_series, pattern):
    return code_series.apply(lambda c: len(re.findall(pattern, str(c))))

# Metrics
csedm_metrics = {
    "comment_count": count_occurrences(csedm_code, r'//.*'),
    "quote_count": count_occurrences(csedm_code, r"'"),
    "if_count": count_occurrences(csedm_code, r"if\("),
    "in_count": count_occurrences(csedm_code, r"\(in"),
    "or_operator_count": count_occurrences(csedm_code, r"\|\|"),  # NEW
}

llm_metrics = {
    "comment_count": count_occurrences(llm_code, r'//.*'),
    "quote_count": count_occurrences(llm_code, r"'"),
    "if_count": count_occurrences(llm_code, r"if\("),
    "in_count": count_occurrences(llm_code, r"\(in"),
    "or_operator_count": count_occurrences(llm_code, r"\|\|"),  # NEW
}

# Run t-tests for each metric
print("=== T-Test Results ===")
for key in csedm_metrics:
    t_stat, p_val = ttest_ind(csedm_metrics[key], llm_metrics[key], equal_var=False)
    print(f"{key}:\n  T-statistic = {t_stat:.4f}, P-value = {p_val:.4e}")
