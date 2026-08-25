@echo off
setlocal

set "SETUP_DIR=%~dp0"
set "MINICONDA_INSTALLER=%SETUP_DIR%Miniconda3-latest-Windows-x86_64.exe"

REM Prefer existing Miniconda, then Anaconda
if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" (
    echo [OK] Found Miniconda.
    exit /b 0
)

if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    echo [OK] Found Anaconda.
    exit /b 0
)

if exist "%LOCALAPPDATA%\Miniconda3\Scripts\activate.bat" (
    echo [OK] Found Miniconda.
    exit /b 0
)

if exist "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" (
    echo [OK] Found Anaconda.
    exit /b 0
)

REM Also check if conda is available in PATH
where conda >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Conda is already available in PATH.
    exit /b 0
)

echo [INFO] Conda was not found.
echo [INFO] Installing Miniconda.

set "MINICONDA_DIR=%USERPROFILE%\Miniconda3"

if not exist "%MINICONDA_INSTALLER%" (
    echo [INFO] Miniconda installer not found in setup folder.
    echo [INFO] Downloading Miniconda...
    powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%MINICONDA_INSTALLER%'"
)

if not exist "%MINICONDA_INSTALLER%" (
    echo [ERROR] Failed to download Miniconda installer.
    exit /b 1
)

echo [INFO] Installing Miniconda silently to:
echo %MINICONDA_DIR%

"%MINICONDA_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%MINICONDA_DIR%

if not exist "%MINICONDA_DIR%\Scripts\activate.bat" (
    echo [ERROR] Miniconda installation failed.
    exit /b 1
)

echo [OK] Miniconda installed successfully.
exit /b 0
