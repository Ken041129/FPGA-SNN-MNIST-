# 虛擬環境:
mkdir snn_mnist_fpga  
cd snn_mnist_fpga  
python -m venv venv  
venv\Scripts\activate  
pip install torch torchvision snntorch matplotlib numpy  
python -c "import torch; print(torch.__version__); import snntorch; print('snnTorch OK')"
# 執行檔案:
python day1_tensor.py
