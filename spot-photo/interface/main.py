from ml_logic.data import load_X_pred

X_pred = load_X_pred(bucket_name = 'bucket_image_flickr30k',
                file_name = 'X_pred_caption_0_to_1000.csv')
print(X_pred.shape)
