from spot_photo.ml_logic.data import load_X_pred, make_corpus, encode_X_pred, load_pickle
from spot_photo.ml_logic.model import load_sentence_similarity_model, embedding_query,\
    compute_similarity, load_captionning_model, predict_step
from spot_photo.ml_logic.results import show_results
from google.oauth2 import service_account
from google.cloud import storage
import pickle

# X_pred = load_X_pred(bucket_name = 'bucket_image_flickr30k',
#                 file_name = 'X_pred_caption_0_to_1000.csv')
# X_pred_embeddings = load_pickle()
# print('✅ X_pred loaded')

# corpus_X_pred = make_corpus(X_pred)
# print('✅ list corpus_X_pred done')

# model = load_sentence_similarity_model(model_name='all-mpnet-base-v2')
# print('✅ model loaded')

# X_pred_embeddings = encode_X_pred(model, corpus_X_pred)
# print('✅ X_pred encoded')

# query_embedding = embedding_query(model, query='dogs playing outsite')
# print('✅ query encoded')

# list_of_image_name = compute_similarity(query_embedding, X_pred_embeddings, k=5)
# print('✅ list of image name obtained')

# show_results(list_of_image_name)
# print('✅ images showed')

def caption_new_images(folder = 'our_full_dataset/'):
    client = storage.Client()
    bucket_name = 'bucket_image_flickr30k'

    bucket = client.get_bucket(bucket_name)
    list_of_blob = list(client.list_blobs(bucket_name, prefix=folder))
    #print(list_of_blob[99:200])
    model, feature_extractor, tokenizer = load_captionning_model()
    print('✅ model loaded')
    start = 4639  # début des cations à faire
    end = 9079 # fin des captions à faire
    step = 10 # nb de photo à prendre pour chaque step
    nb_step = (end - start) // step  #
    rest = (end - start) % nb_step  #
    captions = []
    for n in range (nb_step):
        pred = predict_step(model, feature_extractor, tokenizer, list_of_blob[start+(step*(n)):start+(step*(n+1))])
        captions.extend(pred)
        print(f'✅ predict effectué sur images {start+(step*(n))} à {start+(step*(n+1))}')
        print(f"{len(captions)} captions effectuées")
    if rest != 0 :
        pred = predict_step(model, feature_extractor, tokenizer, list_of_blob[end-rest:])
        captions.extend(pred)
    print('✅ predict terminé')
    with open(f"captions_our_dataset_{start}_{end}.pkl", "wb") as f:
       pickle_captions = pickle.dump(captions, f)

    print('✅ pickle created')


caption_new_images()
