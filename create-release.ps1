#!/usr/bin/env pwsh
<#
.SYNOPSIS
    CELLEX GitHub Release Creator
    Automatically creates GitHub releases with versioned models and documentation

.DESCRIPTION
    This script creates GitHub releases for the Cellex Cancer Detection System.
    It packages the trained model with version naming, includes essential documentation,
    and creates a complete release package for deployment.

.PARAMETER Version
    The version number for the release (e.g., "2.3.0")

.PARAMETER ModelPath
    Path to the model file to include (optional, auto-detects best model)

.PARAMETER Draft
    Create as draft release (optional)

.PARAMETER Prerelease
    Mark as prerelease (optional)

.EXAMPLE
    .\create-release.ps1 -Version "2.3.0"
    
.EXAMPLE
    .\create-release.ps1 -Version "2.3.0" -ModelPath "models/best_model_epoch_2.pth" -Draft

.NOTES
    Requires: GitHub CLI (gh), PowerShell 5.1+
    Author: Cellex Development Team
    Date: October 3, 2025
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Version,
    
    [Parameter(Mandatory=$false)]
    [string]$ModelPath,
    
    [Parameter(Mandatory=$false)]
    [switch]$Draft,
    
    [Parameter(Mandatory=$false)]
    [switch]$Prerelease
)

# Configuration
$ProjectName = "Cellex"
$RepoOwner = "juliuspleunes4"
$RepoName = "cellex"
$TempDir = "temp_release_$Version"

# Colors for output
$ColorSuccess = "Green"
$ColorInfo = "Cyan"
$ColorWarning = "Yellow"
$ColorError = "Red"

function Write-ColorOutput {
    param($Text, $Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

function Test-Prerequisites {
    Write-ColorOutput "🔍 Checking prerequisites..." $ColorInfo
    
    # Check if GitHub CLI is installed - try multiple locations
    $ghPaths = @(
        "gh",  # In PATH
        "C:\Program Files\GitHub CLI\gh.exe",  # Standard install location
        "C:\Program Files (x86)\GitHub CLI\gh.exe",  # 32-bit install
        "$env:LOCALAPPDATA\GitHubCLI\gh.exe"  # Local install
    )
    
    $ghFound = $false
    $ghPath = $null
    
    foreach ($path in $ghPaths) {
        try {
            if ($path -eq "gh") {
                # Test if gh is in PATH
                $null = Get-Command gh -ErrorAction Stop
                $ghVersion = & gh --version 2>$null
            } else {
                # Test specific path
                if (Test-Path $path) {
                    $ghVersion = & $path --version 2>$null
                }
            }
            
            if ($LASTEXITCODE -eq 0 -and $ghVersion) {
                Write-ColorOutput "✅ GitHub CLI found: $($ghVersion[0])" $ColorSuccess
                $ghFound = $true
                $ghPath = $path
                break
            }
        } catch {
            # Continue to next path
        }
    }
    
    if (-not $ghFound) {
        Write-ColorOutput "❌ GitHub CLI not found in common locations." $ColorError
        Write-ColorOutput "   Checked paths:" $ColorError
        foreach ($path in $ghPaths) {
            Write-ColorOutput "   - $path" $ColorError
        }
        Write-ColorOutput "" $ColorError
        Write-ColorOutput "Solutions:" $ColorInfo
        Write-ColorOutput "1. Add GitHub CLI to PATH: Add 'C:\Program Files\GitHub CLI' to your PATH environment variable" $ColorInfo
        Write-ColorOutput "2. Restart PowerShell after installation" $ColorInfo
        Write-ColorOutput "3. Or download from: https://cli.github.com/" $ColorInfo
        exit 1
    }
    
    # Store the working gh path for later use
    $script:ghExecutable = $ghPath
    
    # Check if authenticated with GitHub
    try {
        if ($ghPath -eq "gh") {
            $user = & gh auth status 2>&1
        } else {
            $user = & $ghPath auth status 2>&1
        }
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Authenticated with GitHub" $ColorSuccess
        } else {
            throw "Not authenticated"
        }
    } catch {
        $authCommand = if ($ghPath -eq "gh") { "gh auth login" } else { "`"$ghPath`" auth login" }
        Write-ColorOutput "❌ Not authenticated with GitHub. Run '$authCommand'" $ColorError
        exit 1
    }
    
    # Validate version format
    if (-not ($Version -match '^\d+\.\d+\.\d+$')) {
        Write-ColorOutput "❌ Invalid version format. Use semantic versioning (e.g., 2.3.0)" $ColorError
        exit 1
    }
    
    Write-ColorOutput "✅ All prerequisites met!" $ColorSuccess
}

function Find-BestModel {
    Write-ColorOutput "🔍 Finding best model..." $ColorInfo
    
    $modelsDir = "models"
    if (-not (Test-Path $modelsDir)) {
        Write-ColorOutput "❌ Models directory not found" $ColorError
        return $null
    }
    
    # Look for best models in order of preference
    $modelCandidates = @(
        "best_model_epoch_*.pth",
        "best_*.pth",
        "*.pth"
    )
    
    foreach ($pattern in $modelCandidates) {
        $models = Get-ChildItem "$modelsDir/$pattern" | Sort-Object LastWriteTime -Descending
        if ($models.Count -gt 0) {
            $selectedModel = $models[0].FullName
            Write-ColorOutput "✅ Found model: $($models[0].Name)" $ColorSuccess
            return $selectedModel
        }
    }
    
    Write-ColorOutput "⚠️ No model files found in models directory" $ColorWarning
    return $null
}

function Get-ReleaseNotes {
    Write-ColorOutput "📝 Extracting release notes from CHANGELOG..." $ColorInfo
    
    $changelogPath = "docs/CHANGELOG.md"
    if (-not (Test-Path $changelogPath)) {
        Write-ColorOutput "⚠️ CHANGELOG.md not found, using default release notes" $ColorWarning
        return "Release $Version of the Cellex Cancer Detection System"
    }
    
    try {
        $content = Get-Content $changelogPath -Raw
        
        # Extract version section
        $versionPattern = "## \[$Version\].*?(?=## \[|$)"
        $matches = [regex]::Matches($content, $versionPattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
        
        if ($matches.Count -gt 0) {
            $releaseNotes = $matches[0].Value.Trim()
            Write-ColorOutput "✅ Extracted release notes from CHANGELOG" $ColorSuccess
            return $releaseNotes
        } else {
            Write-ColorOutput "⚠️ Version $Version not found in CHANGELOG, using default notes" $ColorWarning
            return "Release $Version of the Cellex Cancer Detection System"
        }
    } catch {
        Write-ColorOutput "⚠️ Error reading CHANGELOG: $($_.Exception.Message)" $ColorWarning
        return "Release $Version of the Cellex Cancer Detection System"
    }
}

function Create-ReleasePackage {
    Write-ColorOutput "📦 Creating release package..." $ColorInfo
    
    # Create temporary directory
    if (Test-Path $TempDir) {
        Remove-Item $TempDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $TempDir | Out-Null
    
    # Files to include in release
    $filesToInclude = @(
        @{Source = "README.md"; Target = "README.md"},
        @{Source = "requirements.txt"; Target = "requirements.txt"},
        @{Source = "setup.py"; Target = "setup.py"},
        @{Source = "predict_image.py"; Target = "predict_image.py"},
        @{Source = "train.py"; Target = "train.py"},
        @{Source = "verify_dataset.py"; Target = "verify_dataset.py"},
        @{Source = "download_datasets.ps1"; Target = "download_datasets.ps1"}
    )
    
    # Directories to include
    $dirsToInclude = @(
        "src",
        "config", 
        "docs",
        "tests"
    )
    
    # Copy essential files
    foreach ($file in $filesToInclude) {
        if (Test-Path $file.Source) {
            Copy-Item $file.Source "$TempDir/$($file.Target)"
            Write-ColorOutput "  ✅ Added: $($file.Source)" $ColorSuccess
        } else {
            Write-ColorOutput "  ⚠️ Missing: $($file.Source)" $ColorWarning
        }
    }
    
    # Copy directories
    foreach ($dir in $dirsToInclude) {
        if (Test-Path $dir) {
            Copy-Item $dir "$TempDir/" -Recurse
            Write-ColorOutput "  ✅ Added directory: $dir" $ColorSuccess
        } else {
            Write-ColorOutput "  ⚠️ Missing directory: $dir" $ColorWarning
        }
    }
    
    # Copy and rename model file
    $modelFile = $null
    if ($ModelPath) {
        $modelFile = $ModelPath
    } else {
        $modelFile = Find-BestModel
    }
    
    if ($modelFile -and (Test-Path $modelFile)) {
        $modelExtension = [System.IO.Path]::GetExtension($modelFile)
        $versionedModelName = "cellex-v$Version$modelExtension"
        Copy-Item $modelFile "$TempDir/$versionedModelName"
        Write-ColorOutput "  ✅ Added versioned model: $versionedModelName" $ColorSuccess
    } else {
        Write-ColorOutput "  ⚠️ No model file found to include" $ColorWarning
    }
    
    # Create model info file
    $modelInfoPath = "$TempDir/MODEL_INFO.md"
    $modelInfo = @"
# Cellex Model Information - Version $Version

## Model Details
- **Version**: $Version
- **Architecture**: EfficientNet-B0 (Cellex Enhanced)
- **Parameters**: ~5.0M
- **Input Size**: 224x224x3 RGB images
- **Classes**: 2 (0=Healthy, 1=Cancer)

## Performance Metrics
- **Balanced Accuracy**: 98.45%
- **Healthy Accuracy**: 98.85%
- **Cancer Accuracy**: 98.05%
- **Error Rate**: 1.55%

## Usage
1. Place the model file (cellex-v$Version.*) in the `models/` directory
2. Use `predict_image.py` for single image prediction
3. Use the training pipeline in `train.py` for retraining

## Requirements
See `requirements.txt` for complete dependencies.

## Documentation
- Setup: See `README.md`
- API: See `docs/` directory
- Changelog: See `docs/CHANGELOG.md`

Generated on: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss UTC")
"@
    Set-Content -Path $modelInfoPath -Value $modelInfo
    Write-ColorOutput "  ✅ Added: MODEL_INFO.md" $ColorSuccess
    
    # Create ZIP file
    $zipPath = "cellex-v$Version.zip"
    if (Test-Path $zipPath) {
        Remove-Item $zipPath -Force
    }
    
    Write-ColorOutput "📦 Creating ZIP archive..." $ColorInfo
    Compress-Archive -Path "$TempDir/*" -DestinationPath $zipPath -Force
    Write-ColorOutput "✅ Created release package: $zipPath" $ColorSuccess
    
    return $zipPath
}

function Create-GitHubRelease {
    param($ZipPath, $ReleaseNotes)
    
    Write-ColorOutput "🚀 Creating GitHub release..." $ColorInfo
    
    $releaseTitle = "cellex-v$Version.zip"
    
    # Prepare release command
    $releaseArgs = @(
        "release", "create", "v$Version",
        "--title", $releaseTitle,
        "--notes", $ReleaseNotes
    )
    
    if ($Draft) {
        $releaseArgs += "--draft"
        Write-ColorOutput "📝 Creating as draft release" $ColorInfo
    }
    
    if ($Prerelease) {
        $releaseArgs += "--prerelease"
        Write-ColorOutput "🧪 Marking as prerelease" $ColorInfo
    }
    
    # Add ZIP file to release
    $releaseArgs += $ZipPath
    
    try {
        if ($script:ghExecutable -eq "gh") {
            & gh @releaseArgs
        } else {
            & $script:ghExecutable @releaseArgs
        }
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "🎉 Successfully created release v$Version!" $ColorSuccess
            Write-ColorOutput "🔗 Release URL: https://github.com/$RepoOwner/$RepoName/releases/tag/v$Version" $ColorInfo
        } else {
            throw "GitHub CLI returned error code: $LASTEXITCODE"
        }
    } catch {
        Write-ColorOutput "❌ Failed to create release: $($_.Exception.Message)" $ColorError
        exit 1
    }
}

function Cleanup {
    Write-ColorOutput "🧹 Cleaning up..." $ColorInfo
    
    if (Test-Path $TempDir) {
        Remove-Item $TempDir -Recurse -Force
        Write-ColorOutput "✅ Removed temporary directory" $ColorSuccess
    }
    
    # Optionally keep or remove the ZIP file
    $keepZip = Read-Host "Keep the ZIP file ($($zipPath))? (y/N)"
    if ($keepZip.ToLower() -ne 'y') {
        if (Test-Path $zipPath) {
            Remove-Item $zipPath -Force
            Write-ColorOutput "✅ Removed ZIP file" $ColorSuccess
        }
    } else {
        Write-ColorOutput "✅ ZIP file kept: $zipPath" $ColorInfo
    }
}

# Main execution
try {
    Write-ColorOutput "🚀 CELLEX GITHUB RELEASE CREATOR v1.0" $ColorInfo
    Write-ColorOutput ("=" * 50) $ColorInfo
    Write-ColorOutput "Creating release for version: $Version" $ColorInfo
    Write-ColorOutput ""
    
    # Step 1: Check prerequisites
    Test-Prerequisites
    Write-ColorOutput ""
    
    # Step 2: Get release notes
    $releaseNotes = Get-ReleaseNotes
    Write-ColorOutput ""
    
    # Step 3: Create release package
    $zipPath = Create-ReleasePackage
    Write-ColorOutput ""
    
    # Step 4: Create GitHub release
    Create-GitHubRelease -ZipPath $zipPath -ReleaseNotes $releaseNotes
    Write-ColorOutput ""
    
    # Step 5: Cleanup
    Cleanup
    Write-ColorOutput ""
    
    Write-ColorOutput "🎉 RELEASE CREATION COMPLETED SUCCESSFULLY!" $ColorSuccess
    Write-ColorOutput "✅ Release v$Version is now available on GitHub" $ColorSuccess
    
} catch {
    Write-ColorOutput "💥 Fatal error: $($_.Exception.Message)" $ColorError
    Write-ColorOutput "🔍 Stack trace: $($_.ScriptStackTrace)" $ColorError
    
    # Cleanup on error
    if (Test-Path $TempDir) {
        Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    exit 1
}