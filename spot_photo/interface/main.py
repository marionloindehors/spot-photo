from spot_photo.ml_logic.data import load_X_pred, make_corpus, encode_X_pred
from spot_photo.ml_logic.model import load_sentence_similarity_model, embedding_query, compute_similarity
from spot_photo.ml_logic.results import show_results

X_pred = load_X_pred(bucket_name = 'bucket_image_flickr30k',
                file_name = 'X_pred_caption_0_to_1000.csv')
print('✅ X_pred loaded')

corpus_X_pred = make_corpus(X_pred)
print('✅ list corpus_X_pred done')

model = load_sentence_similarity_model(model_name='clip-ViT-B-32')
print('✅ model loaded')

X_pred_embeddings = encode_X_pred(model, corpus_X_pred)
print('✅ X_pred encoded')

query_embedding = embedding_query(model, query='a man playing football')
print('✅ query encoded')

list_of_image_name = compute_similarity(query_embedding, X_pred_embeddings, k=2)
print('✅ list of image name obtained')

show_results(list_of_image_name)
print('✅ images showed')
