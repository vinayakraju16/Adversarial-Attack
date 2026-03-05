@echo off
chcp 65001 >nul

:: 1. Define your server connection
set SERVER=scr0179@129.120.60.102
set REMOTE_DIR=~/thesis_workspace

echo 🚀 Step 1: Setting up the workspace on the GPU Server...
ssh %SERVER% "mkdir -p %REMOTE_DIR%/models %REMOTE_DIR%/results"

echo 📤 Step 2: Sending all Python scripts to the server...
scp *.py %SERVER%:%REMOTE_DIR%/

echo 🧠 Step 3: Training BERT...
ssh %SERVER% "cd %REMOTE_DIR% && python3 02_finetune_bert.py"

echo ⚔️ Step 4: Running A2T Attack on BERT...
ssh %SERVER% "cd %REMOTE_DIR% && python3 04_attack_bert.py"

echo 🧠 Step 5: Training RoBERTa...
ssh %SERVER% "cd %REMOTE_DIR% && python3 03_finetune_roberta.py"

echo ⚔️ Step 6: Running A2T Attack on RoBERTa...
ssh %SERVER% "cd %REMOTE_DIR% && python3 05_attack_roberta.py"

echo 📥 Step 7: Downloading the trained models and attack results to your local machine...
scp -r %SERVER%:%REMOTE_DIR%/models ./
scp -r %SERVER%:%REMOTE_DIR%/results ./

echo 🧹 Step 8: Cleaning up the server...
ssh %SERVER% "rm -rf %REMOTE_DIR%"

echo ✅ ALL COMPLETE! Your models and CSV results are safely on your laptop.
echo ➡️  You can now run 'python 06_results_analysis.py' locally to view the final stats!