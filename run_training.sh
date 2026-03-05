#!/bin/bash

# 1. Define your variables
SERVER="scr0179@129.120.60.102"
SCRIPT_NAME="02_finetune_bert.py"
MODEL_OUTPUT_FOLDER="final_bert_model"

echo "🚀 Sending code to GPU Server..."
scp $SCRIPT_NAME $SERVER:~/

echo "🧠 Starting training on the GPU..."
ssh $SERVER "python3 $SCRIPT_NAME"

echo "📥 Downloading the trained model to your local folder..."
scp -r $SERVER:~/$MODEL_OUTPUT_FOLDER ./

echo "🧹 Cleaning up the server..."
ssh $SERVER "rm -rf ~/$MODEL_OUTPUT_FOLDER ~/$SCRIPT_NAME"

echo "✅ All done! Your trained model is now in your local 'thesis work' folder."