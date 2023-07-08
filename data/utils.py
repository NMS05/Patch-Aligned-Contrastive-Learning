import torchvision.transforms as T
import open_clip

class prepare_data():
    def __init__(self,):
        self.val_transform = T.Compose([
                        T.ToTensor(),
                        T.Resize((400, 400)),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')

    def preprocess_image(self, image):
        return self.val_transform(image)

    def preprocess_text(self,caption):
        return self.tokenizer(caption)