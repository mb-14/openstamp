#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2 START=800 papermill notebooks/save_hidden_states.ipynb output/save_hs.ipynb
# CUDA_VISIBLE_DEVICES=0,1,2 START=1000 papermill notebooks/save_hidden_states.ipynb output/save_hs.ipynb
# papermill notebooks/merge.ipynb output/merge.ipynb

K=8 NUM_EPOCHS=120 papermill notebooks/kmeans.ipynb output/kmeans8.ipynb
K=4 NUM_EPOCHS=120 papermill notebooks/kmeans.ipynb output/kmeans4.ipynb
