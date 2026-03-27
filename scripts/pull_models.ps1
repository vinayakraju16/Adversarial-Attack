$ErrorActionPreference = "Stop"

$RemoteHost     = "scr0179@129.120.60.102"
$RemoteRepoPath = "/home/UNT/scr0179/thesis_work/Adversarial-Attack"
$LocalRepoPath  = Split-Path -Parent $PSScriptRoot

$models = @("tlink_bert_final", "tlink_roberta_final")

foreach ($model in $models) {
    $localDest = Join-Path $LocalRepoPath "models\$model"
    New-Item -ItemType Directory -Force -Path $localDest | Out-Null
    Write-Host "[INFO] Pulling $model ..."
    scp -r "${RemoteHost}:${RemoteRepoPath}/models/${model}/." $localDest
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to pull $model"
        exit 1
    }
    Write-Host "[OK] $model saved to $localDest"
}

Write-Host ""
Write-Host "[DONE] Models saved to: $LocalRepoPath\models\"
Write-Host "[DONE] Run: python codes/predict.py"
