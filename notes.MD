# Pali Gemma from Scratch

## Table of Contents

- [Vision Transformers](#vision-transformers)
- [Contrastive Learning](#contrastive-learning)
- [Multihead Attention](#multi-head-attention)
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

### Aside: Batch Normalization
- in a linear layer, we have a couple parameters, the first one is input features and the 2nd one is output features
    - the input features basically tells us how many features a single input (x) has and the output features will tell us how many neurons there are in that layer
        - the way this works is that there is 1 output feature produced per neuron and since it's a linear layer, it basically dots the input features with the weights and adds a bias term and that is the output for that single neuron
- the problem is called **covariate shift**
    - when you have an input vector that changes from 1 batch to another in magnitude, then the output will also change in magnitude 
        - so let's say we have the first input with values between 1 and 2, but then the next input vector is much larger, this causes a problem. since the loss is dependent on the output of the layer, having large differences between input vectors will also cause the loss to be large, which in turn will impact the gradient, and then the weights will be impacted and will cause the overall model not to do as well (learns very slowly)
- to get around this issue, we implement batch normalization
    - we compute a statistic for each dimension of each item / class (given that they are represented by a vector)
        - we compute the mean and the variance of that dimension and then we normalize each item by subtracting the mean from it and dividing by the standard deviation: $\hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{\mathrm{Var}[x^{(k)}]}}$
- by doing this, it will make each dimension of each item be distributed like a gaussian with mean 0 and variance of 1
    - this will make the model oscillate less which prevents the loss from being large -> which means model performs well / converge faster
- the only draw back is that if the batch size is small, it won't do very well as it depends on what other items are in that batch

### Aside pt. 2: Layer Normalization
- for layer normalization, instead of computing the statistics along the batch dimensions, we calculate them along the item dimension
    - we compute the mean and variance based on the item dimension, thus we process each item independently

### Aside pt. 3: Skip Connections
- the idea behind skip connections is that it is used to bypass certain layers and directly pass the input to a deeper layer
- the reason we do this is to help address the problem of vanishing / exploding gradients and allows information to flow more easily through the model

### Understanding Encoder
- the input to the encoder portion of the vision transformer are the patch embeddings with the positional encodings
    - this input is first passed through a layer normalization (see above). the output of this layer normalization is saved for a skip connection that we do later
- the output from the layer norm. is sent to the attention mechanism. it takes in 3 inputs: key, query, value
    - the attention mechanism is computed as follows: $output = softmax(\frac{Q \cdot K^T}{\sqrt(d_m)}) \cdot V$
- the output from the self-attention is summed up with the skip connection from the layer normalization
- it is then passed through another layer normalization, which passes the output from that to a MLP (multilayer perceptron)
    - note: an MLP is a list of linear layers
    - the output from this layer normalization is also saved for a skip connection later
- the final step is to a sum with the skip connection from the 2nd layer normalization
- the overall output of the encoder are contexualized patches / embeddings


## Multi-Head Attention
- multi-head attention is the models way of contexualizing the input
    - for example, let's look at the vision transformer: we start with a sequence of patches (patches are represented by a vector)
        - the multi-head attention essentially contexualizes these patches / vectors => this means the output will be of the same dimension as the input
        - however, the key thing to note here is that the embeddings / patches which were given as input now not only have information about themselves, but the surrounding patches as well (they are now contexualized)
- for a language model, it is slightly different:
    - we start with a sequence of tokens as the input (this is not necessarily always true, but for simplicity assume 1 token = 1 word)
    - now what happens is that after being passed through the self-attention, each token no longer only captures information about itself, but it also about all the past tokens (the words that came before it)
- we want the language models to predict the next token correctly and for transformers, they can generate the contexualized embeddings in parallel, what we can also do is compute the loss for each prediction in parallel
    - this is the reason the transformer is so powerful (because it computes all this in parallel), in a single pass it can be trained on how to predict the next token given the previous ones

- **understanding key, query, value**
    - step 1: we are going to convert our input $x$ into keys, queries, and values
        - suppose $x$ has size of (4, 1024), to produce the KVQ matrices, we multiply it with $W_q, W_k, W_v$
            - the size of these other matrices are (1024, 1024), but since we are dealing with multi-head attention, suppose num_heads = 8, then the size would be (1024, 8, 128) 
                - to put this more intuitively, the matrix has 1024 rows, each row has a smaller 1-d vector of size 128 and there are 8 of them
        - after the matrix multiplication, the output dimension of the KVQ matrices is (4, 8, 128)
            - in other words, 4 rows with each row having 8 heads of size 128
        - **note:** each head works with a smaller part of the embedding of each token i.e the first head will only look at the first 128 tokens of the entire sequence and so on and so forth
    - step 2: now we will treat each head independently
        - after transposing, we go from (4, 8, 128) -> (8, 4, 128): this basically has 8 rows (number of heads) and each row has a smaller matrix with 4 tokens and each token has 128 dimensions
            - we do this because now we can deal with each head independently 
    - step 3: calculate the attention for each head in parallel
        - we first compute the transpose of the key matrix and then do matrix multiplication with the query matrix for each head
            - the output will be a matrix which represents the dot product of one token with another token (similarity scores)
            