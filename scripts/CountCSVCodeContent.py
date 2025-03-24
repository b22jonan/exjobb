import pandas as pd
import numpy as np
import re
import os

def analyze_java_code(csv_file, output_csv, i):
    df = pd.read_csv(csv_file)
    column_name = "Code" if "Code" in df.columns else "code" if "code" in df.columns else "Extracted_Code" if "Extracted_Code" in df.columns else None
    
    if column_name is None:
        print("Error: No 'Code' or 'code' column found in the CSV file.")
        return
    
    line_counts = []
    word_counts = []
    comment_counts = []
    comment_lengths = []
    empty_line_counts = []
    
    for code in df[column_name].dropna():
        lines = code.split("\n")
        line_counts.append(len(lines))
        
        words = code.split()
        word_counts.append(len(words))
        
        comments = re.findall(r'//.*', code)
        comment_counts.append(len(comments))
        comment_lengths.append(sum(len(comment) for comment in comments))
        
        empty_lines = len([line for line in lines if not line.strip()])
        empty_line_counts.append(empty_lines)
    
    stats = {
        "Name": [
            csv_file,
            "avg_line_count", "std_line_count", 
            "avg_word_count", "std_word_count", 
            "avg_comment_count", "std_comment_count", 
            "avg_comment_length", "std_comment_length", 
            "avg_empty_line_count", "std_empty_line_count"
        ],
        "Value": [
            i,
            np.mean(line_counts), np.std(line_counts),
            np.mean(word_counts), np.std(word_counts),
            np.mean(comment_counts), np.std(comment_counts),
            np.mean(comment_lengths), np.std(comment_lengths),
            np.mean(empty_line_counts), np.std(empty_line_counts)
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    
    if os.path.exists(output_csv):
        # Append to the existing CSV
        stats_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        # If file doesn't exist, create the file and write the header
        stats_df.to_csv(output_csv, mode='w', header=True, index=False)

    print(f"Results for {csv_file} saved to {output_csv}.")
    return stats_df


analyze_java_code("CSV_files\CodeStates.csv", "scripts/results.csv", 1000)

analyze_java_code("prompting\ChatGPT4o\processed_responses.csv", "scripts/results.csv", 2000)
analyze_java_code("prompting\ChatGPT35\processed_responses.csv", "scripts/results.csv", 3000)
analyze_java_code("prompting\DeepSeek\processed_responses.csv", "scripts/results.csv", 4000)
analyze_java_code("prompting\Qwen\processed_responses.csv", "scripts/results.csv", 5000)

analyze_java_code("ML_models/results/XGBoost_ChatGPT4o/LLM.csv", "scripts/results.csv", 1001)
analyze_java_code("ML_models/results/XGBoost_ChatGPT4o/Student.csv", "scripts/results.csv", 1002)
analyze_java_code("ML_models/results/XGBoost_ChatGPT35/LLM.csv", "scripts/results.csv", 1003)
analyze_java_code("ML_models/results/XGBoost_ChatGPT35/Student.csv", "scripts/results.csv", 1004)
analyze_java_code("ML_models/results/XGBoost_DeepSeek/LLM.csv", "scripts/results.csv", 1005)
analyze_java_code("ML_models/results/XGBoost_DeepSeek/Student.csv", "scripts/results.csv", 1006)
analyze_java_code("ML_models/results/XGBoost_Qwen/LLM.csv", "scripts/results.csv", 1007)
analyze_java_code("ML_models/results/XGBoost_Qwen/Student.csv", "scripts/results.csv", 1008)
analyze_java_code("ML_models/results/SVM_ChatGPT4o/LLM.csv", "scripts/results.csv", 1009)
analyze_java_code("ML_models/results/SVM_ChatGPT4o/Student.csv", "scripts/results.csv", 1010)
analyze_java_code("ML_models/results/SVM_ChatGPT35/LLM.csv", "scripts/results.csv", 1011)
analyze_java_code("ML_models/results/SVM_ChatGPT35/Student.csv", "scripts/results.csv", 1012)
analyze_java_code("ML_models/results/SVM_DeepSeek/LLM.csv", "scripts/results.csv", 1013)
analyze_java_code("ML_models/results/SVM_DeepSeek/Student.csv", "scripts/results.csv", 1014)
analyze_java_code("ML_models/results/SVM_Qwen/LLM.csv", "scripts/results.csv", 1015)
analyze_java_code("ML_models/results/SVM_Qwen/Student.csv", "scripts/results.csv", 1016)
analyze_java_code("ML_models/results/NN_ChatGPT4o/LLM.csv", "scripts/results.csv", 1017)
analyze_java_code("ML_models/results/NN_ChatGPT4o/Student.csv", "scripts/results.csv", 1018)
analyze_java_code("ML_models/results/NN_ChatGPT35/LLM.csv", "scripts/results.csv", 1019)
analyze_java_code("ML_models/results/NN_ChatGPT35/Student.csv", "scripts/results.csv", 1020)
analyze_java_code("ML_models/results/NN_DeepSeek/LLM.csv", "scripts/results.csv", 1021)
analyze_java_code("ML_models/results/NN_DeepSeek/Student.csv", "scripts/results.csv", 1022)
analyze_java_code("ML_models/results/NN_Qwen/LLM.csv", "scripts/results.csv", 1023)
analyze_java_code("ML_models/results/NN_Qwen/Student.csv", "scripts/results.csv", 1024)
analyze_java_code("ML_models/results/RandomForest_ChatGPT4o/misclassified_LLM_all.csv", "scripts/results.csv", 1025)
analyze_java_code("ML_models/results/RandomForest_ChatGPT4o/misclassified_Student_all.csv", "scripts/results.csv", 1026)
analyze_java_code("ML_models/results/RandomForest_ChatGPT35/misclassified_LLM_all.csv", "scripts/results.csv", 1027)
analyze_java_code("ML_models/results/RandomForest_ChatGPT35/misclassified_Student_all.csv", "scripts/results.csv", 1028)
analyze_java_code("ML_models/results/RandomForest_DeepSeek/misclassified_LLM_all.csv", "scripts/results.csv", 1029)
analyze_java_code("ML_models/results/RandomForest_DeepSeek/misclassified_Student_all.csv", "scripts/results.csv", 1030)
analyze_java_code("ML_models/results/RandomForest_Qwen/misclassified_LLM_all.csv", "scripts/results.csv", 1031)
analyze_java_code("ML_models/results/RandomForest_Qwen/misclassified_Student_all.csv", "scripts/results.csv", 1032)
analyze_java_code("ML_models/results/adaBoost_ChatGPT4o/misclassified_LLM_all.csv", "scripts/results.csv", 1033)
analyze_java_code("ML_models/results/adaBoost_ChatGPT4o/misclassified_Student_all.csv", "scripts/results.csv", 1034)
analyze_java_code("ML_models/results/adaBoost_ChatGPT35/misclassified_LLM_all.csv", "scripts/results.csv", 1035)
analyze_java_code("ML_models/results/adaBoost_ChatGPT35/misclassified_Student_all.csv", "scripts/results.csv", 1036)
analyze_java_code("ML_models/results/adaBoost_DeepSeek/misclassified_LLM_all.csv", "scripts/results.csv", 1037)
analyze_java_code("ML_models/results/adaBoost_DeepSeek/misclassified_Student_all.csv", "scripts/results.csv", 1038)
analyze_java_code("ML_models/results/adaBoost_Qwen/misclassified_LLM_all.csv", "scripts/results.csv", 1039)
analyze_java_code("ML_models/results/adaBoost_Qwen/misclassified_Student_all.csv", "scripts/results.csv", 1040)
analyze_java_code("ML_models/results/LightGBM_ChatGPT4o/misclassified_LLM_all.csv", "scripts/results.csv", 1041)
analyze_java_code("ML_models/results/LightGBM_ChatGPT4o/misclassified_Student_all.csv", "scripts/results.csv", 1042)
analyze_java_code("ML_models/results/LightGBM_ChatGPT35/misclassified_LLM_all.csv", "scripts/results.csv", 1043)
analyze_java_code("ML_models/results/LightGBM_ChatGPT35/misclassified_Student_all.csv", "scripts/results.csv", 1044)
analyze_java_code("ML_models/results/LightGBM_DeepSeek/misclassified_LLM_all.csv", "scripts/results.csv", 1045)
analyze_java_code("ML_models/results/LightGBM_DeepSeek/misclassified_Student_all.csv", "scripts/results.csv", 1046)
analyze_java_code("ML_models/results/LightGBM_Qwen/misclassified_LLM_all.csv", "scripts/results.csv", 1047)
analyze_java_code("ML_models/results/LightGBM_Qwen/misclassified_Student_all.csv", "scripts/results.csv", 1048)



