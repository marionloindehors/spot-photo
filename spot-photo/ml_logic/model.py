from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

import torch





# DEF A FUNCTION TO LOAD CAPTIONING MODEL

def load_captionning_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    return X_pred #CSV WITH CAPTIONING

#EMBEDDING X_PRED :
#return X_pred_embeddings

# DEF A FUNCTION TO LOAD SENTENCE SIMILARITY MODEL
def load_sentence_similarity_model():
    # Instanciate Model
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

def embedding_query (query):
    # Embedding query
    query_embedding = model.encode(query)
    return query_embedding

def compute_similarity(query_embedding):
    hits = util.semantic_search(query_embedding, X_pred_embeddings, top_k=5)
    hits_sorted = sorted(hits[0], key = lambda ele: ele['score'], reverse=True)

    # Create list of images result index
    list_of_index = []
    for hit in hits_sorted :
        list_of_index.append(hit['corpus_id'])

    # Create list of images name
    list_of_image_name = []
    for i in list_of_index :
        list_of_image_name.append(os.environ ...data['image_name'][i])

    return list_of_image_name



# DEF A FUNCTION TO LOAD TEXT TO IMAGE MODEL * IF BETTER
