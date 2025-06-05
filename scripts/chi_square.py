import os
import re
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# === CONFIG ===
data_folder = "ML_models/code_similarity/csv_files_llm_not_in_use"
output_folder = "charts"

# Make output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === 1. LOAD + LABEL DATA ===
all_data = []

for filename in os.listdir(data_folder):
    if filename.endswith(".csv") and filename.startswith("updated_"):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath)
        
        # Classification label
        if "misclassified" in filename:
            df["classification"] = "Incorrect"
        elif "classified" in filename:
            df["classification"] = "Correct"
        else:
            continue
        
        # Extract ML_model and LLM from filename
        match = re.search(r"updated_(?:mis)?classified_LLM_(.*?)_(.*?)\.csv", filename)
        if match:
            df["ML_model"] = match.group(1)
            df["LLM"] = match.group(2)
        
        all_data.append(df[["PromptType", "classification", "ML_model", "LLM"]])

full_df = pd.concat(all_data, ignore_index=True)

# === 2. CONTINGENCY TABLE + CHI-SQUARE + CRAMÉR'S V ===
contingency_table = pd.crosstab(full_df["PromptType"], full_df["classification"])
chi2, p, dof, expected = chi2_contingency(contingency_table)
n = contingency_table.to_numpy().sum()
k = min(contingency_table.shape)
cramers_v = np.sqrt(chi2 / (n * (k - 1)))

print("\n=== OVERALL CHI-SQUARE TEST ===")
print("Contingency Table:")
print(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p:.2e}")
print(f"Cramér's V: {cramers_v:.4f}")
print("Result:", "Reject null (prompt type affects detectability)" if p < 0.05 else "Fail to reject null")

# === SAVE OVERALL CHI-SQUARE RESULTS ===
overall_results = []

overall_results.append("=== OVERALL CHI-SQUARE TEST ===\n")
overall_results.append("Contingency Table:\n")
overall_results.append(contingency_table.to_string())
overall_results.append(f"\n\nChi-Square Statistic: {chi2:.4f}")
overall_results.append(f"Degrees of Freedom: {dof}")
overall_results.append(f"P-value: {p:.2e}")
overall_results.append(f"Cramér's V: {cramers_v:.4f}")
overall_results.append(
    "Result: Reject null (prompt type affects detectability)" if p < 0.05
    else "Result: Fail to reject null"
)

with open(os.path.join(output_folder, "overall_chi_test.txt"), "w") as f:
    f.write("\n".join(overall_results))


# === 3. STACKED BAR PLOT ===
fig, ax = plt.subplots(figsize=(10, 6))
contingency_table.plot(kind="bar", stacked=True, ax=ax)
plt.title("Classification Outcome by Prompt Type")
plt.xlabel("Prompt Type")
plt.ylabel("Number of Samples")
plt.legend(title="Classification")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "classification_by_prompt.png"))
plt.close()

# === 4. MISCLASSIFICATION RATE PER PROMPT ===
contingency_table["Total"] = contingency_table.sum(axis=1)
contingency_table["MisclassRate"] = contingency_table["Incorrect"] / contingency_table["Total"]
misclass_df = contingency_table[["Incorrect", "Total", "MisclassRate"]].sort_values("MisclassRate", ascending=False)
misclass_df.to_csv(os.path.join(output_folder, "misclassification_rates.csv"))

print("\n=== MISCLASSIFICATION RATE BY PROMPT ===")
print(misclass_df)

# === 5. CHI-SQUARE BY LLM + ML MODEL (Save results to text file) ===
results = []

results.append("\n=== CHI-SQUARE BY LLM ===")
for llm, group in full_df.groupby("LLM"):
    table = pd.crosstab(group["PromptType"], group["classification"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        results.append(f"{llm}: Not enough data for test.")
        continue
    chi2, p, dof, _ = chi2_contingency(table)
    results.append(f"{llm}: Chi2={chi2:.2f}, p={p:.2e}")

results.append("\n=== CHI-SQUARE BY ML MODEL ===")
for model, group in full_df.groupby("ML_model"):
    table = pd.crosstab(group["PromptType"], group["classification"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        results.append(f"{model}: Not enough data for test.")
        continue
    chi2, p, dof, _ = chi2_contingency(table)
    results.append(f"{model}: Chi2={chi2:.2f}, p={p:.2e}")

# Save to text file
with open(os.path.join(output_folder, "per_model_chi_results.txt"), "w") as f:
    f.write("\n".join(results))

print(f"\n All results saved to the '{output_folder}/' directory.")
