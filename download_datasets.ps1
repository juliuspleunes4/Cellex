# =============================================================================
# CELLEX DATASET DOWNLOAD SCRIPT
# =============================================================================
# PowerShell script for downloading medical datasets outside VS Code
# Run this if the dataset download crashes due to size limits in VS Code
# 
# Usage: .\download_datasets.ps1
# =============================================================================

# Colors for output
$Green = "`e[32m"
$Blue = "`e[34m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$Reset = "`e[0m"

# Header
Write-Host "${Blue}🏥 CELLEX MEDICAL AI - DATASET DOWNLOAD${Reset}" -ForegroundColor Blue
Write-Host "${Blue}=============================================${Reset}" -ForegroundColor Blue
Write-Host ""

# Check if we're in the right directory
if (!(Test-Path "main.py") -or !(Test-Path "config/config.yaml")) {
    Write-Host "${Red}❌ Error: Please run this script from the Cellex project root directory${Reset}" -ForegroundColor Red
    Write-Host "${Yellow}   Expected files: main.py, config/config.yaml${Reset}" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment is activated
if (!$env:VIRTUAL_ENV) {
    Write-Host "${Yellow}⚠️  Warning: Virtual environment not detected${Reset}" -ForegroundColor Yellow
    Write-Host "${Yellow}   Attempting to activate .venv...${Reset}" -ForegroundColor Yellow
    
    if (Test-Path ".venv/Scripts/Activate.ps1") {
        & .\.venv\Scripts\Activate.ps1
        Write-Host "${Green}✅ Virtual environment activated${Reset}" -ForegroundColor Green
    } else {
        Write-Host "${Red}❌ Error: Virtual environment not found${Reset}" -ForegroundColor Red
        Write-Host "${Yellow}   Please run: python setup.py${Reset}" -ForegroundColor Yellow
        exit 1
    }
}

# Check Kaggle credentials
Write-Host "${Blue}🔑 Checking Kaggle API credentials...${Reset}" -ForegroundColor Blue
$kaggleJson = "$env:USERPROFILE\.kaggle\kaggle.json"

if (!(Test-Path $kaggleJson)) {
    Write-Host "${Red}❌ Error: Kaggle credentials not found${Reset}" -ForegroundColor Red
    Write-Host "${Yellow}   Please follow these steps:${Reset}" -ForegroundColor Yellow
    Write-Host "${Yellow}   1. Go to kaggle.com → Your Account → API → Create New API Token${Reset}" -ForegroundColor Yellow
    Write-Host "${Yellow}   2. Download kaggle.json${Reset}" -ForegroundColor Yellow
    Write-Host "${Yellow}   3. Place it at: $kaggleJson${Reset}" -ForegroundColor Yellow
    exit 1
}

Write-Host "${Green}✅ Kaggle credentials found${Reset}" -ForegroundColor Green

# Test Kaggle API
Write-Host "${Blue}🧪 Testing Kaggle API connection...${Reset}" -ForegroundColor Blue
try {
    $kaggleTest = python -c "import kaggle; print('OK')" 2>$null
    if ($kaggleTest -eq "OK") {
        Write-Host "${Green}✅ Kaggle API working${Reset}" -ForegroundColor Green
    } else {
        throw "API test failed"
    }
} catch {
    Write-Host "${Red}❌ Error: Kaggle API not working${Reset}" -ForegroundColor Red
    Write-Host "${Yellow}   Please check your kaggle.json file and internet connection${Reset}" -ForegroundColor Yellow
    exit 1
}

# Show dataset information
Write-Host ""
Write-Host "${Blue}📊 VERIFIED CANCER DETECTION DATASETS TO DOWNLOAD${Reset}" -ForegroundColor Blue
Write-Host "${Blue}===============================================${Reset}" -ForegroundColor Blue
Write-Host "${Yellow}1. Chest CT-Scan Cancer Detection${Reset}"
Write-Host "   • Source: mohamedhanyyy/chest-ctscan-images"
Write-Host "   • Size: ~0.8 GB"
Write-Host "   • Samples: 1,000+ CT scans with tumor annotations"
Write-Host "   • Classes: Cancer, Normal"
Write-Host ""
Write-Host "${Yellow}2. Lung Cancer Histopathological Images${Reset}"
Write-Host "   • Source: andrewmvd/lung-and-colon-cancer-histopathological-images"
Write-Host "   • Size: ~1.2 GB"
Write-Host "   • Samples: 25,000+ lung cancer tissue images"
Write-Host "   • Classes: Benign tissue, Adenocarcinoma, Squamous cell carcinoma"
Write-Host ""
Write-Host "${Yellow}3. Brain Tumor Classification (MRI)${Reset}"
Write-Host "   • Source: sartajbhuvaji/brain-tumor-classification-mri"
Write-Host "   • Size: ~0.2 GB"
Write-Host "   • Samples: 3,264+ brain MRI scans"
Write-Host "   • Classes: No Tumor, Glioma, Meningioma, Pituitary"
Write-Host ""
Write-Host "${Yellow}4. Skin Cancer Classification${Reset}"
Write-Host "   • Source: kmader/skin-cancer-mnist-ham10000"
Write-Host "   • Size: ~2.5 GB"
Write-Host "   • Samples: 10,015+ dermatoscopic images"
Write-Host "   • Classes: 7 types including Melanoma, Basal cell carcinoma"
Write-Host ""
Write-Host "${Red}⚠️  TOTAL SIZE: ~4.7 GB${Reset}" -ForegroundColor Red
Write-Host ""

# Confirmation prompt
$response = Read-Host "${Yellow}Do you want to proceed with download? (y/N)${Reset}"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "${Yellow}Download cancelled by user${Reset}" -ForegroundColor Yellow
    exit 0
}

# Check available disk space
Write-Host "${Blue}💾 Checking disk space...${Reset}" -ForegroundColor Blue
$currentPath = Get-Location
$drive = Split-Path -Qualifier $currentPath
$freeSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$drive'").FreeSpace / 1GB

Write-Host "${Blue}Available space: $([math]::Round($freeSpace, 1)) GB${Reset}" -ForegroundColor Blue

if ($freeSpace -lt 50) {
    Write-Host "${Red}⚠️  Warning: Less than 50 GB available${Reset}" -ForegroundColor Red
    $continueAnyway = Read-Host "${Yellow}Continue anyway? (y/N)${Reset}"
    if ($continueAnyway -ne "y" -and $continueAnyway -ne "Y") {
        Write-Host "${Yellow}Download cancelled - insufficient space${Reset}" -ForegroundColor Yellow
        exit 1
    }
}

# Start download
Write-Host ""
Write-Host "${Green}🚀 STARTING DATASET DOWNLOAD${Reset}" -ForegroundColor Green
Write-Host "${Green}=============================${Reset}" -ForegroundColor Green
Write-Host ""

# Record start time
$startTime = Get-Date
Write-Host "${Blue}Start time: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))${Reset}" -ForegroundColor Blue
Write-Host ""

# Run the download command with error handling
try {
    Write-Host "${Blue}Executing: python main.py --mode download${Reset}" -ForegroundColor Blue
    Write-Host ""
    
    # Run with real-time output
    $process = Start-Process -FilePath "python" -ArgumentList "main.py", "--mode", "download" -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        Write-Host ""
        Write-Host "${Green}🎉 DOWNLOAD COMPLETED SUCCESSFULLY!${Reset}" -ForegroundColor Green
        Write-Host "${Green}===================================${Reset}" -ForegroundColor Green
        Write-Host "${Green}End time: $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))${Reset}" -ForegroundColor Green
        Write-Host "${Green}Duration: $($duration.ToString('hh\:mm\:ss'))${Reset}" -ForegroundColor Green
        Write-Host ""
        Write-Host "${Blue}Next steps:${Reset}" -ForegroundColor Blue
        Write-Host "${Blue}1. Start training: python main.py --mode train${Reset}" -ForegroundColor Blue
        Write-Host "${Blue}2. View config: notepad config/config.yaml${Reset}" -ForegroundColor Blue
        Write-Host "${Blue}3. Check data: ls data/processed${Reset}" -ForegroundColor Blue
        
    } else {
        throw "Python process failed with exit code: $($process.ExitCode)"
    }
    
} catch {
    Write-Host ""
    Write-Host "${Red}❌ DOWNLOAD FAILED${Reset}" -ForegroundColor Red
    Write-Host "${Red}=================${Reset}" -ForegroundColor Red
    Write-Host "${Red}Error: $_${Reset}" -ForegroundColor Red
    Write-Host ""
    Write-Host "${Yellow}Troubleshooting:${Reset}" -ForegroundColor Yellow
    Write-Host "${Yellow}1. Check internet connection${Reset}" -ForegroundColor Yellow
    Write-Host "${Yellow}2. Verify Kaggle credentials: $kaggleJson${Reset}" -ForegroundColor Yellow
    Write-Host "${Yellow}3. Check disk space (need ~50 GB)${Reset}" -ForegroundColor Yellow
    Write-Host "${Yellow}4. Try running: python main.py --mode download --verbose${Reset}" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "${Blue}For support: https://github.com/juliuspleunes4/cellex/issues${Reset}" -ForegroundColor Blue
    exit 1
}

Write-Host ""
Write-Host "${Green}✅ Script completed successfully!${Reset}" -ForegroundColor Green