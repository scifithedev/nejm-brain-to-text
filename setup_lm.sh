#!/bin/bash

# Ensure that the script is run from the root directory of the project
if [ ! -f "setup_lm.sh" ]; then
    echo "This script must be run from the root directory of the project."
    exit 1
fi

# ensure that the language_model/runtime/server/x86/build directory does not exist
if [ -d "language_model/runtime/server/x86/build" ]; then
    echo "The language_model/runtime/server/x86/build directory already exists. Please remove it before running this script."
    exit 1
fi

# ensure that the language_model/runtime/server/x86/fc_base directory does not exist
if [ -d "language_model/runtime/server/x86/fc_base" ]; then
    echo "The language_model/runtime/server/x86/fc_base directory already exists. Please remove it before running this script."
    exit 1
fi

# make sure CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "CMake is not installed. Please install CMake >= 3.14 before running this script with 'sudo apt-get install cmake'."
    exit 1
fi

# make sure gcc is installed
if ! command -v gcc &> /dev/null; then
    echo "GCC is not installed. Please install GCC >= 10.1 before running this script with 'sudo apt-get install build-essential'."
    exit 1
fi

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create (or reuse) conda environment with Python 3.9
if conda info --envs | awk '{print $1}' | grep -q "^b2txt25_lm$"; then
    echo "Conda environment 'b2txt25_lm' already exists — reusing it."
else
    conda create -n b2txt25_lm python=3.9 -y
fi

# Activate the environment
conda activate b2txt25_lm

# Upgrade pip
pip install --upgrade pip

# Install additional packages
echo "Installing Python packages (CPU-friendly)..."

# Decide which torch wheel to install: prefer CPU wheel when no NVIDIA GPU detected
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA=1
fi

# Base package list (remove GPU-only packages like bitsandbytes)
PKGS=(
    redis==5.0.6
    jupyter==1.1.1
    numpy==1.24.4
    matplotlib==3.9.0
    scipy==1.11.1
    scikit-learn==1.6.1
    tqdm==4.66.4
    g2p_en==2.1.0
    omegaconf==2.3.0
    huggingface-hub==0.23.4
    transformers==4.40.0
    tokenizers==0.19.1
    accelerate==0.33.0
)

if [ "$HAS_NVIDIA" -eq 1 ]; then
    echo "NVIDIA GPU detected; installing standard torch. If you want a specific CUDA build, adjust the script."
    PKGS+=(torch==1.13.1)
else
    echo "No NVIDIA GPU detected; installing CPU-only PyTorch wheel."
    # Use the official CPU wheel channel for PyTorch
    pip install --index-url https://download.pytorch.org/whl/cpu torch==1.13.1+cpu || pip install torch==1.13.1
fi

# Install the remainder of the packages
pip install "${PKGS[@]}"

# cd to the language model directory and install the language model
cd language_model/runtime/server/x86
python setup.py install

# cd back to the root directory
cd ../../../..

echo
echo "Setup complete! Verify it worked by activating the conda environment with the command 'conda activate b2txt25_lm'."
echo
