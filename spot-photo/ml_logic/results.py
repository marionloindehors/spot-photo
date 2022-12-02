import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def show_results(list_of_image_name):
    # Create list of blobs
    blob_l =[]
    for image in list_of_image_name :
        blob_l.append(bucket.get_blob(f"flickr30k_images/{image}"))
    for blob in blob_l:
        img = Image.open(BytesIO(blob.download_as_bytes()))

    rows = len(list_of_image_name)
    columns = 1
    for  x in range(rows):
        blob_n = blob_l[x]
        img = Image.open(BytesIO(blob_n.download_as_bytes()))
        display(img)
        print(f"Résultat n°{x+1}")
