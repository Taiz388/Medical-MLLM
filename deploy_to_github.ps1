<#
.SYNOPSIS
    自动化将当前文件夹上传到 GitHub 的脚本。
    
.INSTRUCTIONS
    1. 在 VS Code 中打开此文件。
    2. 修改下方的 $RepoUrl 变量为您新建的 GitHub 仓库地址。
    3. 在终端中运行此脚本: .\deploy_to_github.ps1
#>

# --- 配置区域 (每次用于新项目时，请修改此处的仓库地址) ---
# 请将下面的链接替换为您的目标 GitHub 仓库地址
$RepoUrl = "https://github.com/Taiz388/Medical-MLLM.git" 

$BranchName = "main"

# 获取用户输入的提交信息（如果不输入则使用默认值）
$InputMessage = Read-Host "请输入本次更新说明 (直接回车将使用默认值 'Auto-update')"
if ([string]::IsNullOrWhiteSpace($InputMessage)) {
    $CommitMessage = "Auto-update from cleanup script"
} else {
    $CommitMessage = $InputMessage
}
# ----------------------------------------------------

Write-Host "Starting GitHub upload process..." -ForegroundColor Cyan

# 1. 检查是否初始化了 Git，如果没有则初始化
if (-not (Test-Path ".git")) {
    Write-Host "Initializing new Git repository..."
    git init
    # 确保分支名为 main
    git branch -M $BranchName
}

# 2. 配置或更新远程仓库地址
$existingRemote = git remote get-url origin 2>$null
if (-not $existingRemote) {
    Write-Host "Adding remote origin: $RepoUrl"
    git remote add origin $RepoUrl
} elseif ($existingRemote -ne $RepoUrl) {
    Write-Host "Updating remote origin to: $RepoUrl"
    git remote set-url origin $RepoUrl
}

# 3. 添加当前目录下的所有文件
Write-Host "Adding all files..."
git add .

# 4. 检查是否有文件需要提交
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "Committing changes..."
    git commit -m "$CommitMessage"
    
    # 5. 推送到 GitHub
    Write-Host "Pushing to GitHub ($BranchName)..."
    try {
        git push -u origin $BranchName
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Success! Files uploaded to GitHub." -ForegroundColor Green
        } else {
            Write-Host "Push failed. Please checks errors above." -ForegroundColor Red
        }
    } catch {
        Write-Host "An error occurred during push." -ForegroundColor Red
    }
} else {
    Write-Host "No changes detected. Nothing to commit." -ForegroundColor Yellow
}

Write-Host "Done."
