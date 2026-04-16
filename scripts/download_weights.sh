#!/bin/bash

mkdir weights
cd ./weights

# SALAD (~ 350 MiB)
echo "Downloading SALAD weights..."
SALAD_URL="https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
curl -L "$SALAD_URL" -o "./dino_salad.ckpt"

# DINO (~ 340 MiB)
echo "Downloading DINO weights..."
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# DBoW (~ 40 MiB of tar.gz, ~145 MiB of txt)
echo "Downloading DBoW weights..."
(wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz) & wait
tar -xzvf ORBvoc.txt.tar.gz
rm ORBvoc.txt.tar.gz

# Pi3 (~ 3.8 GiB)
echo "Downloading Pi3 weights..."
Pi3_URL="https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors"
curl -L "$Pi3_URL" -o "./pi3.safetensors"

# Pi3X (~ 5.2 GiB)
echo "Downloading Pi3 weights..."
Pi3_URL="https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors"
curl -L "$Pi3_URL" -o "./pi3X.safetensors"

# you will see 5 files under `./weights` when finished
# - pi3.safetensors
# - pi3X.safetensors
# - dino_salad.ckpt             
# - dinov2_vitb14_pretrain.pth  
# - ORBvoc.txt
