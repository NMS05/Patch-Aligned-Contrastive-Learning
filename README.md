# Patch-Aligned-Contrastive-Learning

This repo provides a PyTorch implementation of the paper [Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive
Learning](https://arxiv.org/pdf/2212.04994.pdf), published in CVPR-23' by Meta AI.

Recently there is an increased research interest in **"Open Vocabulary Detection/Segmentation"** systems, that leverage the pre-trained vision-language models like CLIP to perform open-set localization. Pre-training of contrastive models like CLIP primaily focus on "Alignment" (associating the picture of a dog to the english word dog), yet ignore "Grounding" (where the dog is present in the whole picture). This paper proposes a contrastive learning method to perform "Alignment and Grounding" during the pre-training stage itself, thus making the model more "groundable".

## Results

+ Activation map for the prompt "a picture of a cat."
  
<img src="https://assets3.thrillist.com/v1/image/3053693/516x516/flatten;scale;matte=ffffff=center;jpeg_quality=70.jpg" width="350" height="350"> <img src="https://github.com/NMS05/Patch-Aligned-Contrastive-Learning/blob/main/results/cat.png">

+ Activation map for the prompt "a picture of two dogs running."
  
<img src="https://www.dogingtonpost.com/wp-content/uploads/2018/03/dogpark_feature-min.jpg" width="350" height="350"> <img src="https://github.com/NMS05/Patch-Aligned-Contrastive-Learning/blob/main/results/dog.png">

## Directory Structure

+ **data/**
  - image_caption_data.py = a PyTorch Dataset class for MS-COCO that retuns a Image and its tokenized caption or noun-phrase (uses Spacy) as a tensor. 
  - utils.py = preprocesses image and caption for inference.
+ **data/**
  - model.py = contains the PACL model with a novel projection layer and the CLIP loss function.
+ train_pacl.py = train the PACL model and save weights.
+ pacl_inference.py = perform single image inference using the pre-trained CLIP model.
