from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch





# DEF A FUNCTION TO LOAD CAPTIONING MODEL

def load_captionning_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


# DEF A FUNCTION TO LOAD SENTENCE SIMILARITY MODEL

# DEF A FUNCTION TO LOAD TEXT TO IMAGE MODEL * IF BETTER
