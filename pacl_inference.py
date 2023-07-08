import torch
import torch.nn as nn
import torchvision.transforms as T

from data.utils import prepare_data
from model.pacl import open_clip_pacl

from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np


"""
prepare model
"""
device = torch.device("cuda:0")

model = open_clip_pacl()
model_weights = model.state_dict()
saved_weights = torch.load('pacl.pth')
for name in model_weights:
    model_weights[name] = saved_weights['module.' + name]
model.load_state_dict(model_weights)
for p in model.parameters(): p.requires_grad = False
model.to(device)


print("\n\n\n===================================================================================\n\n")


"""
prepare data and get feature projections
"""
with torch.no_grad():
    process = prepare_data()
    img_link = "https://assets3.thrillist.com/v1/image/3053693/516x516/flatten;scale;matte=ffffff=center;jpeg_quality=70.jpg"
    # img_link = "https://www.dogingtonpost.com/wp-content/uploads/2018/03/dogpark_feature-min.jpg"

    image = Image.open(requests.get(img_link, stream=True).raw).convert('RGB')
    image = process.preprocess_image(image).unsqueeze(0)
    image = image.to(device)
    visual_proj = model.forward_visual(image)
    # print(image_projection.shape)

    caption = 'a picture of a cat.'
    # caption = "a picture of two dogs running."

    caption = process.preprocess_text(caption)
    caption = caption.to(device)
    text_proj = model.forward_text(caption)
    # print(text_projections.shape)

    """
    get patch similarities
    """
    similarity_scores = model.patch_alignment(visual_proj, text_proj)
    print(similarity_scores.shape)


similarity_scores = torch.reshape(similarity_scores,(1,25,25))
display_transforms = T.Compose([
    T.GaussianBlur(kernel_size=3)
])
similarity_scores = display_transforms(similarity_scores)
similarity_scores = torch.reshape(similarity_scores,(25,25))

similarity_scores = similarity_scores.cpu().numpy()
plt.imshow(similarity_scores)
plt.show()
