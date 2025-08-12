#!/usr/bin/env zsh
export MPLBACKEND=Agg
# expects the CODESIGN_IDENTITY in the .env file
if [[ -f .env ]]; then
  export $(grep -v '^#' .env | xargs)
fi

rm -rf ./dist && rm -rf ./build

set -e                               # abort on first error

# Create the environment
conda create --name pseg-mac-build python=3.11 -y

# Source conda's init script (works for zsh)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the newly‑created environment
conda activate pseg-mac-build

sleep 5

# Build the Streamlit desktop app inside the env
export BUILD_SOX=1

pip install -r ./Mac/macos-requirements.txt
streamlit-desktop-app build ./app.py \
  --name PetroSegV2 \
  --pyinstaller-options --clean --target-architecture=arm64 --windowed --additional-hooks-dir=./hooks \
  --collect-submodules skimage --collect-submodules sklearn --collect-submodules keras  --collect-submodules threadpoolctl \
  --exclude-module scipy.special._cdflib --exclude-module torch.utils.tensorboard --exclude-module expecttest --noconfirm --noconsole \
  -i ./icons/petroseg.ico

# Deactivate the env before running native macOS tools
conda deactivate

# macOS signing
codesign --deep --force --options=runtime \
  --entitlements ./Mac/entitlements.plist \
  --sign "$CODESIGN_IDENTITY" \
  --timestamp ./dist/PetroSegV2.app

# Clean up the environment
conda remove --name pseg-mac-build --all -y