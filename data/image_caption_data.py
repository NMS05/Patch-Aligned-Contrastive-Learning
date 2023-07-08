from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pycocotools.coco import COCO

from PIL import Image
import spacy
import random
import open_clip

class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, apply_transform=False, img_size=400):

        # chunk for original COCO dataloader
        self.root_dir = root_dir
        self.train_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        T.RandomHorizontalFlip(0.5),
                        T.ColorJitter(brightness=.2, hue=.1),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.val_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.apply_transform = apply_transform
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # chunk for noun phrase extraction -->> creating a prompt from template
        self.template = [
            'a picture of {}.',
            'itap of {}.',
            'a photograph of {}.',
            'this picture contains {}.',
            'a good photo of {}.'
        ]
        self.nlptk = spacy.load("en_core_web_sm")

        # chunk for tokenization
        self.open_clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        
        # chunk for original COCO dataloader
        coco = self.coco
        img_id = self.ids[index]
        caption = coco.imgToAnns[img_id][0]['caption'] # a python string

        img_info = coco.loadImgs(img_id)[0]
        img_path = f"{self.root_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        if self.apply_transform == True:
            image = self.train_transform(image) # tensor of shape = [3, H, W]
        else:
            image = self.val_transform(image)

        # chunk for noun phrase extraction -->> creating a prompt from template
        processed_text = self.nlptk(caption) 
        all_noun_phrases = [chunk.text.lower() for chunk in processed_text.noun_chunks]
        random_template = random.choice(self.template)

        # use original caption 50% of the time
        nounphrase_or_full_caption = random.choice([0,1])

        if len(all_noun_phrases) != 0 and nounphrase_or_full_caption == 0:
            random_noun_phrase = random.choice(all_noun_phrases)
            single_noun_phrase_per_img = random_template.format(random_noun_phrase)
        else:
            single_noun_phrase_per_img = caption
        
        tokenized_phrase = self.open_clip_tokenizer(single_noun_phrase_per_img).squeeze()
        return image, tokenized_phrase

# # # Define the paths to the dataset and annotations
# data_dir = "/home/Dataset/Visual_Recognition/MSCOCO/val2017/"
# annotation_file = "/home/Dataset/Visual_Recognition/MSCOCO/annotations/captions_val2017.json"

# # # Create the dataset and dataloader
# coco_dataset = CocoDataset(data_dir, annotation_file)
# coco_loader = DataLoader(coco_dataset, batch_size=64, shuffle=True)

# for image, caption in coco_loader:
#     print(image.shape, caption.shape)
#     break
