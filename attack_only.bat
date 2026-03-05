@echo off
chcp 65001 >nul

:: Define your server connection
set SERVER=scr0179@129.120.60.102
set REMOTE_DIR=~/thesis_workspace

echo 🚀 Step 1: Setting up the workspace on the GPU Server...
ssh %SERVER% "mkdir -p %REMOTE_DIR%/models %REMOTE_DIR%/results"

echo 📤 Step 2: Uploading trained models and attack scripts (this might take a minute)...
scp -r ./models %SERVER%:%REMOTE_DIR%/
scp 04_attack_bert.py 05_attack_roberta.py %SERVER%:%REMOTE_DIR%/

echo ⚔️ Step 3: Running A2T Attack on BERT...
ssh %SERVER% "cd %REMOTE_DIR% && python3 04_attack_bert.py"

echo ⚔️ Step 4: Running A2T Attack on RoBERTa...
ssh %SERVER% "cd %REMOTE_DIR% && python3 05_attack_roberta.py"

echo 📥 Step 5: Downloading the attack results to your local machine...
scp -r %SERVER%:%REMOTE_DIR%/results ./

echo 🧹 Step 6: Cleaning up the server...
ssh %SERVER% "rm -rf %REMOTE_DIR%"

echo ✅ ATTACKS COMPLETE! 
echo ➡️  You can now run 'python 06_results_analysis.py' locally to view the final stats!