from erlport.erlterms import Atom

from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

import torch
from PIL import Image

if torch.backends.mps.is_available():
    torch.set_default_device('mps')

if torch.cuda.is_available():
    torch.set_default_device('cuda')


class Embedder:
    def __init__(self):
        self.model = None
        self.text_model = None
        self.processor = None

    def load_model(self, name):
        name = name.decode("utf-8")
        try:
            self.model = CLIPVisionModelWithProjection.from_pretrained(name)
            self.text_model = CLIPTextModelWithProjection.from_pretrained(name)
            self.processor = CLIPProcessor.from_pretrained(name)
            return Atom(b"ok")
        except Exception as err:
            print(err)
            return Atom(b"error")

embedder = Embedder()

def load_model(name):
    return embedder.load_model(name)

def get_text_embedding(text):
    text = text.decode("utf-8")
    inputs = embedder.processor(text = text, return_tensors="pt", padding=True)
    text_outputs = embedder.text_model(**inputs)
    text_embeds = text_outputs.text_embeds.tolist()[0]
    return text_embeds

def get_image_embedding(image_path):
    try:
        path = image_path.decode("utf-8")
        image = Image.open(path)
        inputs = embedder.processor(images=image, return_tensors="pt", padding=True)
        image_outputs = embedder.model(**inputs)
        image_embeds = image_outputs.image_embeds.tolist()[0]
        return image_embeds
    except Exception as err:
        print(err)
        return False
