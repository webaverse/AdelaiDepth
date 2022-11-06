
#!/bin/bash

wget -O ./res101.pth https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download

pip3 install ipykernel matplotlib opencv-python
pip3 install torch==1.6.0 torchvision==0.7.0
#cudatoolkit==10.2
pip3 install flask
