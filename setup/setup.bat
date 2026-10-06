@echo off
echo [[ Starting Windows Naoqi Environment Setup ]]

:: 1. Create and activate the conda env
call conda create --name nao -c conda-forge python=2.7 pillow -y
call conda activate nao

:: 2. Create directory for SDK
mkdir "%CONDA_PREFIX%\Lib\site-packages\pynaoqi"

:: 3. Check for the Windows SDK zip file
dir "%USERPROFILE%\Downloads\pynaoqi-python2.7-*.zip" >nul 2>&1
if errorlevel 1 (
    echo [[ ERROR: Could not find the pynaoqi .zip SDK file in your Downloads folder! ]]
    echo Please download the Windows SDK from the Maxtronics Developer Center and leave it in your Downloads folder.
    pause
    exit /b 1
)

:: 4. Extract the Windows SDK zip file
echo [[ Extracting SDK from Downloads... ]]
tar -xf "%USERPROFILE%\Downloads\pynaoqi-python2.7-*.zip" -C "%CONDA_PREFIX%\Lib\site-packages\pynaoqi" --strip-components=1

:: 5. Bind environment variable
echo [[ Binding PYTHONPATH... ]]
call conda env config vars set PYTHONPATH="%CONDA_PREFIX%\Lib\site-packages\pynaoqi\lib"

:: 6. Reload environment
call conda deactivate
call conda activate nao

echo [[ Setup Complete! Your NAOqi environment is ready. ]]
echo To run the robot actuation layer: 'conda activate nao' then 'python scripts\nao_tts.py'
pause