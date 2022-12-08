from spot_photo.ml_logic.data import load_X_pred, make_corpus, encode_X_pred, load_pickle,\
    upload_file, extend_pickle
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

def caption_new_images(folder = 'flickr30k_images'):
    client = storage.Client()
    bucket_name = 'bucket_image_flickr30k'

    bucket = client.get_bucket(bucket_name)
    list_of_blob = list(client.list_blobs(bucket_name, prefix=f"{folder}/"))
    print(f"{len(list_of_blob)}  images à captionner")
    model, feature_extractor, tokenizer = load_captionning_model()
    print('✅ model loaded')
    start = 6000  # début des cations à faire
    end = 10000 # fin des captions à faire
    step = 10 # nb de photo à prendre pour chaque step
    nb_step = (end - start) // step  #
    rest = (end - start) % nb_step  #

    captions = []
    for n in range (nb_step):

        pred = predict_step(model, feature_extractor, tokenizer, list_of_blob[start+(step*(n)):start+(step*(n+1))])
        captions.extend(pred)
        print(f'✅ predict effectué sur images {start+step*n} à {start+(step*(n+1))}')
        print(f"{len(captions)} captions effectuées")

    if rest != 0 :
        pred = predict_step(model, feature_extractor, tokenizer, list_of_blob[end-rest:end])
        captions.extend(pred)
        print(f'✅ predict effectué sur images {end-rest} à {end}')
        print(f"{len(captions)} captions effectuées")
    print('✅ predict terminé')
    with open(f"{folder}_{start}_{end}.pkl", "ab") as f:
        pickle.dump(captions, f)

    print('✅ pickle created')

#caption_new_images()

#upload_file('flickr30k_images_6000_10000.pkl', 'flickr30k_images_6000_10000.pkl', bucket_name="bucket_image_flickr30k")
#extend_pickle('flickr30k_images_0_1500.pkl', 'flickr30k_images_6000_10000.pkl')

def embbed_pickle():
    model = load_sentence_similarity_model(model_name="all-mpnet-base-v2")
    print('model loaded')
    list_caption = load_pickle(file_name="our_full_dataset_full_caption_all_mpnet_base_v2.pkl")
    print('pickle ouvert')
    X_pred_corpus = make_corpus(list_caption)
    print('corpus créé')
    print(len(X_pred_corpus))
    list_embbede = encode_X_pred(model, X_pred_corpus)
    print('corpus embbedé')
    print(len(list_embbede))
    list_img_caption = []
    for path, tensor in zip(list_caption, list_embbede) :
        list_img_caption.append((path[0], tensor))
    print('liste créée')
    print(len(list_img_caption))
    print(len(list_img_caption[0][1]))
    with open(f"our_full_dataset_full_embedded_allmpnetbasev2.pkl", "wb") as f:
        pickle.dump(list_img_caption, f)
    upload_file("our_full_dataset_full_embedded_allmpnetbasev2.pkl", "our_full_dataset_full_embedded_allmpnetbasev2.pkl", bucket_name="bucket_image_flickr30k")
    print('file uploaded')
    return list_img_caption

#embbed_pickle()
list_caption = load_pickle(file_name="all_dataset_clipvitb32.pkl")
print(list_caption[0][1].shape)
#extend_pickle('flickr30k_images_embedded_0_10000_allmpnetbasev2.pkl', "our_full_dataset_full_embedded_allmpnetbasev2.pkl")
