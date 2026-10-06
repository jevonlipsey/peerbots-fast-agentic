#!/bin/bash

echo "[[ Starting Naoqi Environment Setup ]]"

# 1. Initialize Conda for the script environment (CRITICAL FOR SCRIPTS)
eval "$(conda shell.bash hook)"

# 2. OS-Specific Setup (Mac vs Linux)
OS_TYPE=$(uname)

if [ "$OS_TYPE" = "Darwin" ]; then
    echo "[[ macOS Detected: Installing Rosetta and osx-64 Conda env ]]"
    # Suppress error if Rosetta is already installed
    softwareupdate --install-rosetta --agree-to-license 2>/dev/null || true
    CONDA_SUBDIR=osx-64 conda create --name nao -c conda-forge python=2.7 pillow -y
elif [ "$OS_TYPE" = "Linux" ]; then
    echo "[[ Linux Detected: Creating standard Conda env ]]"
    conda create --name nao -c conda-forge python=2.7 pillow -y
else
    echo "Unsupported OS for this script. Use setup.bat for Windows."
    exit 1
fi

# 3. Activate the environment
conda activate nao

# 4. Create the directory and extract the SDK
echo "[[ Extracting SDK... ]]"
mkdir -p "$CONDA_PREFIX/lib/python2.7/site-packages/pynaoqi"
LATEST_SDK=$(ls -t ~/Downloads/*pynaoqi*.tar.gz 2>/dev/null | head -n 1)

if [ -z "$LATEST_SDK" ]; then
    echo "[[ ERROR: Could not find the pynaoqi .tar.gz SDK file in your Downloads folder! ]]"
    echo "Please download the Mac/Linux SDK from the Maxtronics Developer Center and leave the .tar.gz file in your Downloads folder."
    exit 1
fi

tar -xzf "$LATEST_SDK" -C "$CONDA_PREFIX/lib/python2.7/site-packages/pynaoqi" --strip-components=1

# 5. Bind Environment Variables
echo "[[ Binding Environment Variables... ]]"
conda env config vars set PYTHONPATH="$CONDA_PREFIX/lib/python2.7/site-packages/pynaoqi/lib/python2.7/site-packages"
conda env config vars set QI_SDK_PREFIX="$CONDA_PREFIX/lib/python2.7/site-packages/pynaoqi"

if [ "$OS_TYPE" = "Darwin" ]; then
    conda env config vars set DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib/python2.7/site-packages/pynaoqi/lib"
fi

# 6. Reload to lock in variables
conda deactivate
conda activate nao

# 7. Mac Only: Re-link C++ Libraries
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "[[ Re-linking Apple Silicon C++ Libraries... ]]"
    find "$CONDA_PREFIX/lib/python2.7/site-packages/pynaoqi" -type f \( -name "*.so" -o -name "*.dylib" \) -exec install_name_tool -change /Library/Frameworks/Python.framework/Versions/2.7/Python "$CONDA_PREFIX/lib/libpython2.7.dylib" {} 2>/dev/null \;
fi

echo "[[ Setup Complete! Your NAOqi environment is ready. ]]"
echo "To run the robot actuation layer: 'conda activate nao' then 'python scripts/nao_tts.py'"