#@title Download libraries and clone git project(run only for first time)
# %cd /content/
# !git clone https://github.com/aim-uofa/AdelaiDepth
# !wget -O /content/AdelaiDepth/res101.pth https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download
import sys
import os
#sys.path.append('/usr/local/lib/python3.7/site-packages')
sys.path.append('./LeReS')
#os.environ["PYTHONPATH"] += (":/usr/local/lib/python3.7/site-package")
os.environ["PYTHONPATH"] += (":./LeReS")
# !echo "$PYTHONPATH"
# from IPython.display import clear_output
# clear_output()














#@title Clean output directory
# %%shell
# if [ -e /content/AdelaiDepth/LeReS/test_images ]; then
#   if [ -n "`ls -A /content/AdelaiDepth/LeReS/test_images/`" ]; then
#     rm -rf /content/AdelaiDepth/LeReS/test_images/*
#     echo "Cleaned"
#   fi
# elif [ ! -e /content/AdelaiDepth/LeReS/test_images ]; then
#   mkdir /content/AdelaiDepth/LeReS/test_images
#   mkdir /content/AdelaiDepth/LeReS/test_images/outputs
#   echo "Created directories and cleaned"
# fi















#@title Upload images(png, jpg works fine, also you can upload multiple images)
# %cd /content/AdelaiDepth/LeReS/test_images/
# from google.colab import files
# image = files.upload()
#for n in image.keys():
#  print("{name} succesfully uploaded!".format(name = n))


















#@title Run network
# !echo "$PYTHONPATH"
# %cd /content/AdelaiDepth
python3 LeReS/tools/test_depth.py --load_ckpt res101.pth --backbone resnext101
# from IPython.display import clear_output
# clear_output()
# !echo "Done!"



















#@title Download results
# from google.colab import files
# %cd /content/AdelaiDepth/LeReS/test_images/
# !find outputs/ -name "*-depth_raw.png" | zip -r result.zip -@
# files.download("result.zip")

















#@title Check videocard
# !nvidia-smi