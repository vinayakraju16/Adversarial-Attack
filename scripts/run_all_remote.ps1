$ErrorActionPreference = "Stop"

# ============================================================================
# USER-EDITABLE VARIABLES - update these if your host or paths ever change
# ============================================================================
$RemoteHost     = "scr0179@129.120.60.102"
$RemoteRepoPath = "/home/UNT/scr0179/thesis_work/Adversarial-Attack"
$RsyncExe       = "C:\Program Files\Git\usr\bin\rsync.exe"
# ============================================================================

$LocalRepoPath    = Split-Path -Parent $PSScriptRoot
$LocalResultsPath = Join-Path $LocalRepoPath "results"

$RequiredRelativeFiles = @(
    "codes/02_finetune_bert.py",
    "codes/03_finetune_roberta.py",
    "run.py",
    "codes/06_results_analysis.py",
    "scripts/run_pipeline_gpu.sh"
)

$ExcludePatterns = @(
    "models/",
    "results/",
    "results/",
    ".git/",
    "__pycache__/",
    ".cache/",
    "wandb/",
    "runs/",
    "venv/",
    ".venv/"
)

function Fail-Step {
    param([string]$Message, [int]$Code = 1)
    Write-Error $Message
    exit $Code
}

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$FilePath,
        [string[]]$Arguments
    )
    Write-Host "[INFO] $Label"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail-Step "$Label failed with exit code $LASTEXITCODE."
    }
}

function Convert-ToRsyncPath {
    param([string]$PathValue)
    $resolved = (Resolve-Path $PathValue).Path.Replace("\", "/")
    if ($resolved -match "^([A-Za-z]):/(.*)$") {
        $drive = $matches[1].ToLower()
        $rest  = $matches[2]
        return "/$drive/$rest"
    }
    return $resolved
}

# ----------------------------------------------------------------------------
# PRE-FLIGHT CHECKS
# ----------------------------------------------------------------------------

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Fail-Step "ssh not found in PATH. Install OpenSSH client before running this script."
}

if (-not (Test-Path $LocalRepoPath)) {
    Fail-Step "Local repo path does not exist: $LocalRepoPath"
}

foreach ($rel in $RequiredRelativeFiles) {
    $full = Join-Path $LocalRepoPath $rel
    if (-not (Test-Path $full)) {
        Fail-Step "Required file missing locally: $rel - restore it before running the pipeline."
    }
}

Write-Host "[INFO] Verifying SSH connectivity to $RemoteHost ..."
ssh -o BatchMode=yes -o ConnectTimeout=10 $RemoteHost "echo ssh-ok"
if ($LASTEXITCODE -ne 0) {
    Fail-Step "Cannot reach $RemoteHost via SSH. Check: VPN is active, SSH key is loaded, and RemoteHost is correct."
}

Write-Host "[INFO] Verifying remote venv at ~/.venvs/advnlp ..."
ssh -o BatchMode=yes $RemoteHost "test -f ~/.venvs/advnlp/bin/activate"
if ($LASTEXITCODE -ne 0) {
    Fail-Step "Remote venv not found at ~/.venvs/advnlp on $RemoteHost. Create it first: python3 -m venv ~/.venvs/advnlp"
}

Write-Host "[INFO] Ensuring remote repo directories exist ..."
ssh $RemoteHost "mkdir -p '$RemoteRepoPath' '$RemoteRepoPath/results' '$RemoteRepoPath/results/attacks' '$RemoteRepoPath/results/summary'"
if ($LASTEXITCODE -ne 0) {
    Fail-Step "Could not create remote directories under $RemoteRepoPath. Check permissions on the GPU server."
}

New-Item -ItemType Directory -Force -Path $LocalResultsPath | Out-Null

# ----------------------------------------------------------------------------
# PHASE 1 - SYNC LOCAL TO GPU (rsync or tar+scp fallback)
# ----------------------------------------------------------------------------

$UseRsync = Test-Path $RsyncExe

if ($UseRsync) {
    Write-Host "[INFO] Phase 1/3: Syncing local repo to GPU with rsync ..."
    $syncArgs = @("-az", "--delete")
    foreach ($pat in $ExcludePatterns) {
        $syncArgs += "--exclude"
        $syncArgs += $pat
    }
    $localRsyncPath = (Convert-ToRsyncPath $LocalRepoPath).TrimEnd("/") + "/"
    $syncArgs += $localRsyncPath
    $syncArgs += "${RemoteHost}:${RemoteRepoPath}/"
    Invoke-Checked -Label "rsync local repo to GPU" -FilePath $RsyncExe -Arguments $syncArgs
} else {
    Write-Host "[WARN] rsync.exe not found at '$RsyncExe'. Falling back to tar and scp."
    Write-Host "[INFO] Phase 1/3: Syncing local repo to GPU with tar and scp ..."

    $TarFile = Join-Path $env:TEMP "repo_sync.tgz"

    $tarExcludes = $ExcludePatterns | ForEach-Object { "--exclude=./$_" }
    $tarArgs = @("-czf", $TarFile) + $tarExcludes + @("-C", $LocalRepoPath, ".")

    Write-Host "[INFO]   Creating archive: $TarFile"
    & tar @tarArgs
    if ($LASTEXITCODE -ne 0) { Fail-Step "tar archive creation failed. Ensure tar is available (Git Bash or WSL)." }

    Write-Host "[INFO]   Uploading archive to GPU ..."
    scp $TarFile "${RemoteHost}:${RemoteRepoPath}/repo_sync.tgz"
    if ($LASTEXITCODE -ne 0) { Fail-Step "scp upload failed." }

    Write-Host "[INFO]   Extracting archive on GPU ..."
    ssh $RemoteHost "tar -xzf '$RemoteRepoPath/repo_sync.tgz' -C '$RemoteRepoPath' && rm '$RemoteRepoPath/repo_sync.tgz'"
    if ($LASTEXITCODE -ne 0) { Fail-Step "Remote tar extraction failed." }

    Write-Host "[INFO]   Removing local archive ..."
    Remove-Item $TarFile -Force
}

# ----------------------------------------------------------------------------
# PHASE 2 - RUN REMOTE PIPELINE
# ----------------------------------------------------------------------------

Write-Host "[INFO] Phase 2/3: Running GPU pipeline via SSH ..."
ssh $RemoteHost "bash '$RemoteRepoPath/scripts/run_pipeline_gpu.sh'"
if ($LASTEXITCODE -ne 0) { Fail-Step "Remote GPU pipeline failed. Check the SSH output above for the error." }

# ----------------------------------------------------------------------------
# PHASE 3 - PULL RESULTS BACK TO LOCAL
# ----------------------------------------------------------------------------

Write-Host "[INFO] Phase 3/3: Pulling results back to local results/ ..."

if ($UseRsync) {
    $localResultsRsyncPath = (Convert-ToRsyncPath $LocalResultsPath).TrimEnd("/") + "/"
    $pullArgs = @("-az", "--delete", "${RemoteHost}:${RemoteRepoPath}/results/", $localResultsRsyncPath)
    Invoke-Checked -Label "rsync GPU results to local results/" -FilePath $RsyncExe -Arguments $pullArgs
} else {
    Write-Host "[INFO]   Packing remote results/ ..."
    ssh $RemoteHost "tar -czf '$RemoteRepoPath/results_export.tgz' -C '$RemoteRepoPath' results"
    if ($LASTEXITCODE -ne 0) { Fail-Step "Remote results archive creation failed." }

    $ResultsTar = Join-Path $LocalResultsPath "results_export.tgz"
    Write-Host "[INFO]   Downloading results archive ..."
    scp "${RemoteHost}:${RemoteRepoPath}/results_export.tgz" $ResultsTar
    if ($LASTEXITCODE -ne 0) { Fail-Step "scp download of results failed." }

    Write-Host "[INFO]   Extracting results locally ..."
    tar -xzf $ResultsTar -C $LocalResultsPath --strip-components=1
    if ($LASTEXITCODE -ne 0) { Fail-Step "Local extraction of results failed." }

    Remove-Item $ResultsTar -Force
    ssh $RemoteHost "rm -f '$RemoteRepoPath/results_export.tgz'"
}

Write-Host ""
Write-Host "[DONE] Remote pipeline finished."
Write-Host "[DONE] Results pulled to: $LocalResultsPath"
