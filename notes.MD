# Pali Gemma from Scratch

## Table of Contents

- Vision Transformers
- [Contrastive Learning](#contrastive-learning)
- Language Model (Gemma)
- KV-Cache
- Rotary Positional Embedding
- Normalization


## Contrastive Learning
- Within this model, there is a contrastive vision encoder which takes an image as input and converts it into an embedding
    - It converts the image into a series of embeddings
    - the image will be split into patches and each patch will be converted into an embedding
    - these embeddings will be concatenated with the token embeddings and will be sent to the transformer
- We want the embeddings for the image and text to be similar
    - In other words, we want the dot product of the embeddings to be high 
    - similarly, we want the dot product of the embeddings that are not the same / corresponding to be low
- Following this, we want to find a loss function that will encourage the model to learn the embeddings that are similar i.e we want the dot products of the same numbered embeddings to be high and the others to be lower
    - We use cross entropy loss for this
- **pseudocode from CLIP paper**

```python
I_f = image_encoder(I) # convert the image into embeddings
T_f = text_encoder(T) # convert the text into embeddings

I_e = l2_normalize(np.dot(I_f, W_i), axis=1) # we make sure both emebeddings are normalized and have the same dimension
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

logits = np.dot(I_e, T_e.T) * np.exp(t) # compute all the possible dot products

labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0) # we teach the model which items in each row / column are similar and needs to be maximized
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t) / 2
```
- **what is the problem with CLIP**
    - the problem is that we are using cross entropy loss

- **understanding importance of numerical stability for softmax**
    - the problem is that the softmax function is numerically unstable
        - in other words, the exponential function grows very fast and it may not be able to fit into a 32 bit floating-point (in computers)
    - 