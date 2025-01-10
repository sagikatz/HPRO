@echo off
REM Set up and activate a Python virtual environment on Windows

REM Step 1: Check if Python is installed
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python is not installed or not in PATH. Please install Python first.
    exit /b 1
)

REM Step 2: Create a virtual environment in the 'venv' directory
echo Creating a virtual environment...
python -m venv venv

REM Step 3: Activate the virtual environment
echo Activating the virtual environment...
call venv\Scripts\activate

REM Step 4: Upgrade pip in the virtual environment
echo Upgrading pip...
pip install --upgrade pip

REM Step 5: Install dependencies from requirements.txt (if file exists)
IF EXIST requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) ELSE (
    echo No requirements.txt file found. Skipping this step.
)

REM Step 6: Install PyTorch, TorchVision, and TorchAudio with CUDA 11.8 support
echo Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Step 7: Confirm completion
echo Setup completed successfully.
exit /b 0
