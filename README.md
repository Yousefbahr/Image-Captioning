# Image-Captioning
An encoder-decoder model consisting of a Vision Transformer (ViT) for extracting image features with a DistilGPT2 for generating captions.

The model attained a BLEU-1 score of 0.66, indicating a high degree of unigram-level alignment. 

The model was trained on a subset of the [COCO Image Captioning Dataset](https://www.kaggle.com/datasets/nagasai524/mini-coco2014-dataset-for-image-captioning?select=Images), which consists of images and at least 5 captions per image.

<table>
  <tr>
    <td><img src="Images/my_plot3.png" ></td>
    <td><img src="Images/my_plot4.png" ></td>
    <td><img src="Images/my_plot5.png" ></td>
  </tr>
  <tr>
    <td>A white cat sitting in a bathtub in a bathroom.</td>
     <td>A man riding skis down a snow covered slope.</td>
     <td>A zebra grazing on green grass in a field.</td>
  </tr>
 </table>

## Architecture
The image captioning model consists of a Vision Transformer (ViT) encoder and a DistilGPT2 decoder:

- The ViT encoder processes input images and outputs a sequence of visual embeddings representing image features.

- These features are first projected from the encoder's embedding space to a higher-dimensional space, then reduced back down to match the decoder's embedding dimension, using two linear layers, Dropout, and GELU activation.
  
- A learned [SEP] token embedding is appended to distinguish image features from text input.

- The DistilGPT2 decoder then takes the concatenated embeddings (image + [SEP] + text) as input and generates captions token by token in an autoregressive manner.

## Training
A custom attention mask is implemented for the visual features and the [SEP] token.

Data Parallelism is being used to increase the input batch size.

During training:

- Each sample includes an image and a corresponding randomly selected, tokenized caption from a set of captions.

- Captions are passed as inputs and labels (for causal language modeling).

- The ViT encoder extracts visual features, which are then fed into a neural network. These features are concatenated with the [SEP] token and the caption's token embeddings before being passed to the DistilGPT2 decoder.

- The model computes a causal language modeling loss, predicting the next word at each position in the caption.


