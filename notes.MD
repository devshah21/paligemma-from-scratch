# Pali Gemma from Scratch

## Table of Contents

- [Vision Transformers](#vision-transformers)
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
    - what we need to do is make the softmax function numerically stable
        - in order to do this, we do not make the exponential grow to infinity: $\frac{c \cdot e^{a_i}}{c \cdot \sum_{k=1}^{N} e^{a_k}} 
= \frac{e^{\log(c)} \cdot e^{a_i}}{e^{\log(c)} \cdot \sum_{k=1}^{N} e^{a_k}} 
= \frac{e^{a_i + \log(c)}}{\sum_{k=1}^{N} e^{a_k + \log(c)}}$
        - what we can do is that we set $log(c) = -max(a_i)$, in doing so, it will be less likely the exponential will go to infinity

- **normalization factor in the softmax**
    - since we have a matrix of columns and rows which contain the dot products of the word and text embeddings, we cannot go through all the elements since that is very compute heavy
        - instead, since the softmax loss is asymmetric, the normalization (softmax) can be performed independently twice, once across the images and once across texts
    - in the contrastive paper, instead of using the cross-entropy loss, they proposed to use sigmoid loss
        - the main idea is that we don't treat the loss over a column / row, instead we treat each entry in the matrix as a binary classification task
            - in other words, it should either be 0 or 1 and we treat them independently from the other entries. this is done by using the **sigmoid** function
            - since we are treating them independently, we don't have to worry about the complexities of normalization over large matrices, which simplifies the computation and improves numerical stability

- **why use a contrastive vision encoder?**
    - we want the embeddings to be good representations that can be contrasted with the text embeddings
        - in other words, we basically contrast the text with the image
    - another reason we use it is because it's cheaper to train

## Vision Transformers
- the way these transformers work is that we take the image and split it into patches
    - then we extract information using a convolutional and flatten it to produce an embedding of patches (**recall**: when we flatten, we lose positional information)
- then we add positional encodings to the patch embeddings to restore the spaital information 
    - these positional encodings are typically vectors of the same length as the patch embeddings and are added element-wise to each patch embedding
- after getting the positional encodings, we pass it to the transformer and the output we get are the **contextualized embeddings**
    - since the input is an image, the model isn't auto-regressive. this basically means that the model doesn't make predictions sequentially i.e the next prediction does not rely on the previous ones
    - each patch will basically have information about it's position and what is around it in the image