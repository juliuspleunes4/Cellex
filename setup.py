"""
CELLEX CANCER DETECTION SYSTEM - SETUP SCRIPT
=============================================
Professional setup and installation script.
"""

import subprocess
import sys
import os
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    """Print Cellex banner."""
    banner = f"""
{Colors.OKCYAN}
[SYMBOL]
[SYMBOL]                                                                   [SYMBOL]
[SYMBOL]           [MEDICAL] CELLEX CANCER DETECTION SYSTEM [MEDICAL]                    [SYMBOL]
[SYMBOL]                                                                   [SYMBOL]
[SYMBOL]         Advanced AI-Powered Medical Imaging Analysis              [SYMBOL]
[SYMBOL]           Built by AI Scientists for Medical Professionals       [SYMBOL]
[SYMBOL]                                                                   [SYMBOL]
[SYMBOL]
{Colors.ENDC}
    """
    print(banner)


def run_command(command, description):
    """Run a command with error handling."""
    print(f"{Colors.OKBLUE}[PROGRESS] {description}...{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"{Colors.OKGREEN}[SUCCESS] {description} completed successfully{Colors.ENDC}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}[ERROR] {description} failed: {e}{Colors.ENDC}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print(f"{Colors.OKBLUE}[PYTHON] Checking Python version...{Colors.ENDC}")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"{Colors.FAIL}[ERROR] Python 3.8+ required. Current: {version.major}.{version.minor}.{version.micro}{Colors.ENDC}")
        return False
    
    print(f"{Colors.OKGREEN}[SUCCESS] Python {version.major}.{version.minor}.{version.micro} is compatible{Colors.ENDC}")
    return True


def check_pip():
    """Check if pip is available."""
    print(f"{Colors.OKBLUE}[SYMBOL] Checking pip...{Colors.ENDC}")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print(f"{Colors.OKGREEN}[SUCCESS] pip is available{Colors.ENDC}")
        return True
    except subprocess.CalledProcessError:
        print(f"{Colors.FAIL}[ERROR] pip not found{Colors.ENDC}")
        return False


def install_requirements():
    """Install Python requirements."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print(f"{Colors.FAIL}[ERROR] requirements.txt not found{Colors.ENDC}")
        return False
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    command = f"{sys.executable} -m pip install -r requirements.txt"
    return run_command(command, "Installing Python dependencies")


def create_directories():
    """Create necessary directories."""
    print(f"{Colors.OKBLUE}[FOLDER] Creating directories...{Colors.ENDC}")
    
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "logs",
        "results",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print(f"{Colors.OKGREEN}[SUCCESS] Directories created successfully{Colors.ENDC}")
    return True


def setup_kaggle():
    """Guide user through Kaggle setup."""
    print(f"{Colors.OKBLUE}[SYMBOL] Kaggle API Setup{Colors.ENDC}")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print(f"{Colors.OKGREEN}[SUCCESS] Kaggle credentials already configured{Colors.ENDC}")
        return True
    
    print(f"{Colors.WARNING}[WARNING]  Kaggle credentials not found{Colors.ENDC}")
    print("To download datasets, you need to setup Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print(f"4. Place it at: {kaggle_json}")
    
    if os.name != 'nt':  # Not Windows
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
    
    response = input(f"\n{Colors.OKCYAN}Have you set up Kaggle credentials? (y/N): {Colors.ENDC}")
    
    if response.lower() in ['y', 'yes']:
        if kaggle_json.exists():
            print(f"{Colors.OKGREEN}[SUCCESS] Kaggle setup verified{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.WARNING}[WARNING]  kaggle.json still not found at {kaggle_json}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.WARNING}[WARNING]  Kaggle setup skipped - you can set it up later{Colors.ENDC}")
        return False


def create_config():
    """Create default configuration."""
    print(f"{Colors.OKBLUE}[SYMBOL][INFO]  Creating default configuration...{Colors.ENDC}")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Run config creation
    try:
        subprocess.run([sys.executable, "config/config.py"], check=True)
        print(f"{Colors.OKGREEN}[SUCCESS] Configuration created{Colors.ENDC}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}[ERROR] Configuration creation failed: {e}{Colors.ENDC}")
        return False


def test_installation():
    """Test if installation works."""
    print(f"{Colors.OKBLUE}[TEST] Testing installation...{Colors.ENDC}")
    
    # Test imports
    test_commands = [
        "python -c \"import torch; print(f'PyTorch: {torch.__version__}')\""
    ]
    
    for command in test_commands:
        if not run_command(command, f"Testing: {command.split('import')[1].split(';')[0].strip()}"):
            return False
    
    print(f"{Colors.OKGREEN}[SUCCESS] Installation test passed{Colors.ENDC}")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print(f"""
{Colors.HEADER}{Colors.BOLD}
[COMPLETE] CELLEX SETUP COMPLETED SUCCESSFULLY! [COMPLETE]
{Colors.ENDC}

{Colors.OKCYAN}Next Steps:{Colors.ENDC}

1. {Colors.OKGREEN}Download Data:{Colors.ENDC}
   python main.py --mode download
   
2. {Colors.OKGREEN}Train Model:{Colors.ENDC}
   python main.py --mode train
   
3. {Colors.OKGREEN}Make Predictions:{Colors.ENDC}
   python main.py --mode predict --image path/to/xray.jpg
   
4. {Colors.OKGREEN}Run Complete Pipeline:{Colors.ENDC}
   python main.py --mode pipeline

{Colors.OKCYAN}Documentation:{Colors.ENDC}
   [SYMBOL] README.md - Project overview
   [SYMBOL] CHANGELOG.md - Development history
   [SYMBOL][INFO]  config/config.py - Configuration options

{Colors.OKCYAN}Need Help?{Colors.ENDC}
   python main.py --help

{Colors.OKGREEN}Happy detecting! [RESEARCH][MEDICAL]{Colors.ENDC}
    """)


def main():
    """Main setup function."""
    print_banner()
    
    setup_steps = [
        ("Python Version Check", check_python_version),
        ("Pip Check", check_pip),
        ("Install Dependencies", install_requirements),
        ("Create Directories", create_directories),
        ("Setup Kaggle API", setup_kaggle),
        ("Create Configuration", create_config),
        ("Test Installation", test_installation),
    ]
    
    failed_steps = []
    
    print(f"{Colors.HEADER}{Colors.BOLD}[LAUNCH] Starting Cellex Setup Process{Colors.ENDC}\n")
    
    for step_name, step_function in setup_steps:
        print(f"{Colors.OKCYAN}{'='*50}")
        print(f"STEP: {step_name}")
        print(f"{'='*50}{Colors.ENDC}")
        
        if not step_function():
            failed_steps.append(step_name)
            
        print()  # Empty line for spacing
    
    # Summary
    if not failed_steps:
        print_next_steps()
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}[ERROR] Setup completed with errors:{Colors.ENDC}")
        for step in failed_steps:
            print(f"   - {step}")
        print(f"\n{Colors.WARNING}[WARNING]  Please fix the above issues and run setup again{Colors.ENDC}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)