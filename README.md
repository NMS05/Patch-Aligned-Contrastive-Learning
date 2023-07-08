# Patch-Aligned-Contrastive-Learning (PACL)

This repo provides a PyTorch implementation of the paper [Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive
Learning](https://arxiv.org/pdf/2212.04994.pdf), published in CVPR-23' by Meta AI.

Recently there is an increased research interest in **"Open Vocabulary Detection/Segmentation"** systems, that leverage the pre-trained vision-language models like CLIP to perform open-set localization. Pre-training of contrastive models like CLIP primaily focus on "Alignment" (associating the picture of a dog to the english word dog), yet ignore "Grounding" (where the dog is present in the whole picture). This paper proposes a contrastive learning method to perform "Alignment and Grounding" during the pre-training stage itself, which could potentially enhance the performance of open vocabulary detection/segmentation systems.

## Highlights
+ Leverages the pre-trained open_CLIP (laion2b-s34b-b88K) model and fine-tunes it on MS-COCO dataset.
+ In addition to text captions, object-centric noun phrases are extracted using Spacy.
+ Performs training/inference at a higher resolution of 400p (interpolates the position embedding) from the original 224p.
+ Utilizes scaled Sigmoid instead of the original Softmax for patch activations.

# Results

+ Activation map for the prompt **"a picture of a cat."**
  
<img src="https://assets3.thrillist.com/v1/image/3053693/516x516/flatten;scale;matte=ffffff=center;jpeg_quality=70.jpg" width="350" height="350"> <img src="https://github.com/NMS05/Patch-Aligned-Contrastive-Learning/blob/main/results/cat.png">

+ Activation map for the prompt **"a picture of two dogs running."**
  
<img src="https://www.dogingtonpost.com/wp-content/uploads/2018/03/dogpark_feature-min.jpg" width="350" height="350"> <img src="https://github.com/NMS05/Patch-Aligned-Contrastive-Learning/blob/main/results/dog.png">

## Directory Structure

+ **data/**
  - image_caption_data.py = a PyTorch Dataset class for MS-COCO that retuns a Image and its tokenized caption or noun-phrase (uses Spacy) as a tensor. 
  - utils.py = preprocesses image and caption for inference.
+ **data/**
  - model.py = contains the PACL model with a novel projection layer and the CLIP loss function.
+ train_pacl.py = train the PACL model and save weights.
+ pacl_inference.py = perform single image inference using the pre-trained PACL model.

## Limitations
Scale Matters!! - The original paper trained the model on 30M image-text pairs (GCC-3M + GCC-12M + YFCC-15M), while COCO contains only 110k image-text pairs. Hence the resulting activation map contains a lot of artifacts.

## References
+ [Open_CLIP](https://github.com/mlfoundations/open_clip/tree/main/src)
