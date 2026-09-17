param(
    [switch]$RunJobs,
    [switch]$Foreground,
    [string]$LogFile
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not $RunJobs -and -not $Foreground) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    if (-not $LogFile) {
        $LogFile = Join-Path $logDir "cifar10_baselines_$stamp.log"
    }
    $pidFile = Join-Path $logDir "cifar10_baselines.pid"

    $p = Start-Process -FilePath "powershell.exe" `
        -WorkingDirectory $PSScriptRoot `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $PSCommandPath,
            "-RunJobs",
            "-LogFile", $LogFile
        ) `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -Path $pidFile -Value $p.Id -Encoding ascii
    Write-Host "已在后台启动，关闭本窗口不影响运行。"
    Write-Host "PID : $($p.Id)"
    Write-Host "日志: $LogFile"
    Write-Host "停止: Stop-Process -Id $($p.Id)"
    exit 0
}

if (-not $LogFile) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $LogFile = Join-Path $logDir "cifar10_baselines_$stamp.log"
}

Start-Transcript -Path $LogFile -Append | Out-Null
try {
    $env:PYTHONUNBUFFERED = "1"
    $py = "C:\Users\11023\miniconda3\envs\MoE\python.exe"

    $jobs = @(
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "resnet", "--select", "ours", "--merge", "ours", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "resnet", "--select", "posthoc", "--merge", "ours", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "resnet", "--select", "fedrolex", "--merge", "ours", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "resnet", "--select", "snip", "--merge", "ours", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "resnet", "--select", "ours", "--merge", "fedavg", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "resnet", "--select", "ours", "--merge", "fedavgm", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "resnet", "--select", "ours", "--merge", "fedadam", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "vgg11", "--select", "ours", "--merge", "ours", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "vgg11", "--select", "posthoc", "--merge", "ours", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "vgg11", "--select", "fedrolex", "--merge", "ours", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "vgg11", "--select", "snip", "--merge", "ours", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "vgg11", "--select", "ours", "--merge", "fedavg", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "vgg11", "--select", "ours", "--merge", "fedavgm", "--device", "cuda"),
        @("--method", "ours", "--dataset", "cifar10", "--backbone", "vgg11", "--select", "ours", "--merge", "fedadam", "--device", "cuda")
    )

    $i = 0
    foreach ($jobArgs in $jobs) {
        $i += 1
        Write-Host ("`n[{0}/{1}] {2} {3}" -f $i, $jobs.Count, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), ($jobArgs -join " "))
        & $py -u main.py @jobArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Job $i failed with exit code $LASTEXITCODE"
        }
    }
    Write-Host ("`n全部完成: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
}
finally {
    Stop-Transcript | Out-Null
}
