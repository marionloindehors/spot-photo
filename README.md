# Deep Learning project - Bootcamp Le Wagon #1035 
- Spot-photo
- Description: Creation of a photo search engine application using several models of Deep Learning, image recognition and “natural language processing”. Building of an API, a Streamlit interface and deployment on the cloud (Docker, GCP).

# Details
We used several pre-trained models to develop our image search tools and we compared different methods.

The first one consists in labeling our images, then calculating the similarity between our request and these labels, by encoding our texts.

The second method consists of directly encoding our images and calculating the similarity between our encoded query and our encoded images.

Technical stack:
- Python, Pytorch, Sentence Transformers, Transformers
- Models: all-mpnet-base-V2, clip-ViT-v32, ViTFeatureExtractor
- Google Colab, Google Cloud Platform, Docker, Streamlit, GitHub 
