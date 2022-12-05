from spot_photo.ml_logic.data import load_X_pred, make_corpus, encode_X_pred, load_pickle
from spot_photo.ml_logic.model import load_sentence_similarity_model, embedding_query,\
    compute_similarity, load_captionning_model, predict_step
from spot_photo.ml_logic.results import show_results
from google.oauth2 import service_account
from google.cloud import storage

# #X_pred = load_X_pred(bucket_name = 'bucket_image_flickr30k',
# #                file_name = 'X_pred_caption_0_to_1000.csv')
# X_pred_embeddings = load_pickle()
# print('✅ X_pred loaded')

# #corpus_X_pred = make_corpus(X_pred)
# #print('✅ list corpus_X_pred done')

# model = load_sentence_similarity_model(model_name='all-mpnet-base-v2')
# print('✅ model loaded')

# #X_pred_embeddings = encode_X_pred(model, corpus_X_pred)
# #print('✅ X_pred encoded')

# query_embedding = embedding_query(model, query='two dark skin people with a grey umbrella sitting on the street with things to sell')
# print('✅ query encoded')

# list_of_image_name = compute_similarity(query_embedding, X_pred_embeddings, k=5)
# print('✅ list of image name obtained')

# show_results(list_of_image_name)
# print('✅ images showed')

def caption_new_images(folder = 'people_dataset_2054/'):
    credentials = service_account.Credentials.from_service_account_file(
    'possible-aspect-369317-b19475afaf02.json')
    bucket_name = 'bucket_image_flickr30k'
    client = storage.Client(credentials=credentials)

    bucket = client.get_bucket(bucket_name)
    list_of_blob = list(client.list_blobs(bucket_name, prefix=folder))
    print(list_of_blob[:2])
    model, feature_extractor, tokenizer = load_captionning_model()
    print('model loaded')
    pred = predict_step(model, feature_extractor, tokenizer, list_of_blob[:2])

    return pred

print(caption_new_images())
