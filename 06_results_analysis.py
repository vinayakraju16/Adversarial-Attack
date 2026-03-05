import pandas as pd

def analyze_results(csv_path, model_name):
    try:
        df = pd.read_csv(csv_path)
        
        # Count outcomes
        total_attacks = len(df)
        successful_attacks = len(df[df['result_type'] == 'Successful'])
        failed_attacks = len(df[df['result_type'] == 'Failed'])
        skipped = len(df[df['result_type'] == 'Skipped'])
        
        # Calculate Attack Success Rate (ignoring skipped items where the model was already wrong)
        valid_attacks = total_attacks - skipped
        asr = (successful_attacks / valid_attacks) * 100 if valid_attacks > 0 else 0
        
        print(f"\n=== Analysis for {model_name} ===")
        print(f"Total Samples Processed: {total_attacks}")
        print(f"Skipped (Model predicted wrong originally): {skipped}")
        print(f"Failed Attacks (Model defended successfully): {failed_attacks}")
        print(f"Successful Attacks (Model was tricked): {successful_attacks}")
        print(f"--> Attack Success Rate (ASR): {asr:.2f}%")
        
        # Calculate average words changed
        if 'num_words_changed' in df.columns:
            avg_words_changed = df[df['result_type'] == 'Successful']['num_words_changed'].mean()
            print(f"--> Average words changed per successful attack: {avg_words_changed:.1f}")
            
    except FileNotFoundError:
        print(f"\n⚠️ Could not find {csv_path}. Make sure the attack script finished running!")

# Run Analysis
analyze_results("./results/bert_a2t_results.csv", "BERT")
analyze_results("./results/roberta_a2t_results.csv", "RoBERTa")