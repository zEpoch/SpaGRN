#!/usr/bin/env bash
# Run during image construction: bash scripts/install_gpu.sh
set -eo pipefail
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n spagrn-gpu python=3.8 pip -y
conda activate spagrn-gpu
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install --upgrade "pip<25.1" "setuptools<75" wheel
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -c constraints-gpu.txt -e .
python -m pip install -c constraints-gpu.txt ipykernel
python -m pip check
python -c 'import torch; import spagrn; from spagrn.regulatory_network import InferNetwork; print("SpaGRN:", spagrn.__file__); print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPU available:", torch.cuda.is_available())'
python -m unittest discover -s test -p 'test_geary_gpu.py' -v
python -m ipykernel install --name=spagrn-gpu --display-name "Python (spagrn-gpu)"
python -m pip freeze > requirements-installed.txt
