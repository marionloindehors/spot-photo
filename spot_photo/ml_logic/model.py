from sentence_transformers import SentenceTransformer, util
from google.oauth2 import service_account
from google.cloud import storage
import pickle
from spot_photo.ml_logic.data import load_data
from PIL import Image
from io import BytesIO

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer


# ---------------------------------------
# DEF A FUNCTION TO LOAD CAPTIONING MODEL
# ---------------------------------------
def load_captionning_model():
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer


def predict_step(model, feature_extractor, tokenizer, list_of_blob):
    max_length = 20  # Nombre de mots dans la caption
    num_beams = 4  # on ne sait pas ce qu'on sait
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    images = []
    for blob in list_of_blob:
        i_image = Image.open(BytesIO(blob.download_as_bytes()))
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
    print(len(images))
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    # pixel_values = pixel_values.to(device)

    output_ids = model.generate(
        pixel_values, **gen_kwargs
    )  # matrice des captions encodées (?)

    preds = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )  # decode les matrices en mots
    preds = [
        (image.name, pred.strip()) for image, pred in zip(list_of_blob, preds)
    ]  # list de tuple ( nom image,  caption)
    # preds = [ pred.strip() for pred in preds]  #list de chaque caption

    #CREATE PICKLE FROM TUPLE CAPTIONS en local

    with open(f"captions_our_dataset_200.pkl", "wb") as f:
       pickle_captions = pickle.dump(preds, f)

    return pickle_captions


# ------------------------------------------------
# DEF A FUNCTION TO LOAD SENTENCE SIMILARITY MODEL
# -------------------------------------------------
def load_sentence_similarity_model(model_name="all-mpnet-base-v2"):
    # Instanciate Model
    if model_name == "all-mpnet-base-v2":
        model = SentenceTransformer("all-mpnet-base-v2")
    if model_name == "clip-ViT-B-32":
        model = SentenceTransformer("clip-ViT-B-32")
    return model


# --------------------------------
# DEF A FUNCTION TO EMBEDD A QUERY
# --------------------------------
def embedding_query(model, query):
    # Embedding query
    query_embedding = model.encode(query)  # /!\ clip vit return matrice shape 512!!
    return query_embedding


# ---------------------------------------
# DE A FUNCTION TO COMPUTE THE SIMILARITY
# -----------------------------------------
def compute_similarity(query_embedding, X_pred_embeddings, k=2):  # images_embedding

    hits = util.semantic_search(query_embedding, X_pred_embeddings, top_k=k)
    hits_sorted = sorted(hits[0], key=lambda ele: ele["score"], reverse=True)

    # Create list of images result index
    list_of_index = []
    for hit in hits_sorted:
        list_of_index.append(hit["corpus_id"])
    data = load_data()
    # Create list of images name
    list_of_image_name = []
    for i in list_of_index:
        image_na = data["image_name"][i * 5]
        list_of_image_name.append(image_na)
    return list_of_image_name


#    if model == SentenceTransformer('clip-ViT-B-32'):
#         list_cos_scores = []
#         score = util.cos_sim(query_embedding, X_pred_embeddings) #X_pred_embedding Encoded features from images
#         list_cos_scores.append(score)
#         return max(list_cos_scores) #Tensor
