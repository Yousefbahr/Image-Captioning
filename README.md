# Image-Captioning
Combines a Vision Transformer (ViT) encoder for extracting image features with a DistilGPT2 decoder for generating captions.

The model was trained on a subset of the [COCO Image Captioning Dataset](https://www.kaggle.com/datasets/nagasai524/mini-coco2014-dataset-for-image-captioning?select=Images), which consists of images and at least 5 captions per image.

## Architecture
The image captioning model consists of a Vision Transformer (ViT) encoder and a DistilGPT2 decoder:

- The ViT encoder processes input images and outputs a sequence of visual embeddings representing image features.

- These features are first projected from the encoder's embedding space to a higher-dimensional space, then reduced back down to match the decoder's embedding dimension, using two linear layers, Dropout, and GELU activation.
  
- A learned [SEP] token embedding is appended to distinguish image features from text input.

- The DistilGPT2 decoder then takes the concatenated embeddings (image + [SEP] + text) as input and generates captions token by token in an autoregressive manner.

## Training
During training:

- Each sample includes an image and a corresponding randomly selected, tokenized caption from a set of captions.

- Captions are passed as inputs and labels (for causal language modeling).

- The ViT encoder extracts visual features, which are concatenated with the [SEP] token and caption token embeddings.

- The model computes a causal language modeling loss, predicting the next word at each position in the caption.
