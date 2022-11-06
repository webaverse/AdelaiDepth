
#!/bin/bash

wget -O ./res101.pth https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download

conda create --name adelai
conda activate adelai
pip3 install -r requirements.txt
