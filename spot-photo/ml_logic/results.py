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

    # Create figure
    def fig ():
        fig = plt.figure(figsize=(15, 30))

        # Setting values to rows and column variables
        rows = 5
        columns = 1

        # Reading images
        blob0 = blob_l[0]
        blob1 = blob_l[1]
        blob2 = blob_l[2]
        blob3 = blob_l[3]
        blob4 = blob_l[4]

        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)

        # showing image
        #blob0 = blob_l[0]
        plt.imshow(Image.open(BytesIO(blob0.download_as_bytes())))
        plt.axis('off')
        plt.title("Best score")

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)

        # showing image
        #blob1 = blob_l[1]
        plt.imshow(Image.open(BytesIO(blob1.download_as_bytes())))
        plt.axis('off')
        plt.title("Second choice")

        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 3)

        # showing image
        blob2 = blob_l[2]
        plt.imshow(Image.open(BytesIO(blob2.download_as_bytes())))
        plt.axis('off')
        plt.title("Third choice")

        # Adds a subplot at the 4th position
        fig.add_subplot(rows, columns, 4)

        # showing image
        blob3 = blob_l[3]
        plt.imshow(Image.open(BytesIO(blob3.download_as_bytes())))
        plt.axis('off')
        plt.title("Fourth choice")

        # Adds a subplot at the 5th position
        fig.add_subplot(rows, columns, 5)

        # showing image
        blob4 = blob_l[4]
        plt.imshow(Image.open(BytesIO(blob4.download_as_bytes())))
        plt.axis('off')
        plt.title("Fifth choice")

        return fig
    return fig
