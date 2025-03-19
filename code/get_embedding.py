# EXAMPLE USAGE:
# python get_embedding.py configs/default.yaml checkpoints/default-epoch=009.ckpt

import sys
import torch
import pandas as pd
import numpy as np
import yaml
import glob
import os
from tqdm import tqdm

from autoencoder import Autoencoder
from data import make_data

config_path = sys.argv[1]
checkpoint_path = sys.argv[2]

config = yaml.safe_load(open(config_path, "r"))

print("Loading the saved model")
# initialize the autoencoder class
model = Autoencoder(patch_size=config["data"]["patch_size"], **config["autoencoder"])
# tell PyTorch to load the model onto the CPU if no GPU is available
map_location = None if torch.cuda.is_available() else 'cpu'
# load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=map_location)
# load the checkpoint's state_dict into the model
model.load_state_dict(checkpoint["state_dict"])
# put the model in evaluation mode
model.eval()
filepaths = sorted(glob.glob("../data/image_data/*.npz"))

print("Making the patch data")
# receive labeled_indices
images_long, patches, labeled_indices = make_data(patch_size=config["data"]["patch_size"])

print("Obtaining embeddings")
# get the embedding for each patch
embeddings = []  # what we will save
images_embedded = []  # for visualization

for i in tqdm(labeled_indices):
    fp = filepaths[i]  # Get the file path based on the index
    npz_data = np.load(fp)
    key = list(npz_data.files)[0]
    raw_data = npz_data[key]  # Original data (including label column)
    
    if raw_data.shape[1] == 11:  # 带标签的数据
        labels = raw_data[:, -1]  # 最后一列为标签
    else:
        labels = np.zeros(raw_data.shape[0])  # 无标签数据，填充为0
        
    ys = images_long[i][:, 0]
    xs = images_long[i][:, 1]

    # determine the height and width of the image
    miny = min(ys)
    minx = min(xs)
    height = int(max(ys) - miny + 1)
    width = int(max(xs) - minx + 1)

    # to make this faster, we use torch.no_grad() to disable gradient tracking
    with torch.no_grad():
        # get the embedding of array of patches
        emb = model.embed(torch.tensor(np.array(patches[i])))
        # NOTE: if your model is quite big, you may not be able to fit
        # all of the data into the GPU memory at once for inference.
        # In that case, you can loop over smaller bathches of data.

        # in the following line we:
        # - detach the tensor from the computation graph
        # - move it to the cpu
        # - turn it into a numpy array
        emb = emb.detach().cpu().numpy()

    embeddings.append(emb)
    
    # Add a label column and adjust the column order
    embedding_size = config["autoencoder"]["embedding_size"]
    embedding_df = pd.DataFrame(emb, columns=[f"ae{i}" for i in range(embedding_size)])
    embedding_df["y"] = ys
    embedding_df["x"] = xs
    
    # Add label column
    if raw_data.shape[1] == 11:  
        embedding_df["label"] = labels 

    # Adjust column order：y, x, label, ae0-aeN
    cols = ["y", "x", "label"] + [c for c in embedding_df.columns if c not in ["y", "x", "label"]]
    embedding_df = embedding_df[cols]

    # save the embeddings as csv
    filename = os.path.basename(filepaths[i]).replace(".npz", "_ae.csv")
    embedding_df.to_csv(f"../data/{filename}", index=False)

print("Saving the embeddings")

# here is some code to take a look at the embeddings.
# but you should probably just load the csv files in a jupyter notebook
# and visualize there.

# import matplotlib.pyplot as plt
# plt.imshow(images_embedded[0][0])
# plt.show()