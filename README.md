# Introduction to Deep Learning

Introduction to Deep Learning course the Technical University Munich (TUM) [SS 2023](https://niessner.github.io/I2DL/).

## Outline

- Sentiment Analysis using Recurrent Neural Network
- Semantic Segmentation using Convolutional Neural Network
- Facial Keypoint Detection Using Convolutional Neural Network
- Handwritten Digits Classification on MNIST Using Auto-encoder
- Image Classification on CIFAR-10 Using Neural Network
- House Prices Classification Using Logistic Regression
- Dataset and Dataloader Classes

## Sentiment Analysis using Recurrent Neural Network

Sentiment analysis of sample of the `IMDb movie reviews dataset` using `recurrent neural network`. The goal is to classify the reviews into positive or negative.
<p class="aligncenter">
    <img src="exercise_11/images/IMDB.jpg" alt="centered image" />
</p>
<p class="aligncenter">
    <img src="exercise_11/images/examples.png" alt="centered image" />
</p>

### Preprocessing a Text Classification Datase

For this part, the target is to convert text into a sequence of integers. The following steps are performed:
- tokenize input text
- create a vocabulary and include `<eos>` and `<unk>`
- map each token to an index in the vocabulary
- create dataset class
- create dataloader class
- minibatch the data, while padding the shorter sequences with `<eos>` tokens

### Embeddings

In order to express the semantic relations between words, we use an embedding layer will learn the semantic relations between words as the model is trained. Pre-trained embedding vectors such as [word2vec](https://arxiv.org/abs/1301.3781) or [GLoVe](https://nlp.stanford.edu/projects/glove/) could be used as well.

<img src='https://developers.google.com/machine-learning/crash-course/images/linear-relationships.svg'/>

### Recurrent Neural Network

Create a network that applies an embedding layer to input, followed by a RNN layer, and finally a linear layer to obtain the final output.

<p class="aligncenter">
    <img src="exercise_11/images/LSTM.png" alt="centered image" />
</p>


## Semantic Segmentation Using Convolutional Neural Network

Segmentation of images in MSRC-v2 Segmentation Dataset using `convolutional neural network`. The goal is to classify each pixel in the image into one of the 23 classes.

<img src="https://camo.githubusercontent.com/d10b897e15344334e449104a824aff6c29125dc2/687474703a2f2f63616c76696e2e696e662e65642e61632e756b2f77702d636f6e74656e742f75706c6f6164732f646174612f636f636f7374756666646174617365742f636f636f73747566662d6578616d706c65732e706e67">

### Training Pipeline

- initialize train_loader
- hyperparameters selection
- initialize model
- for each epoche, perform the following steps on each batch:
    - forward pass
    - calculate loss
    - backward pass
    - optimizer step
    - track loss and accuracy

## Facial Keypoint Detection Using Convolutional Neural Network

Facial keypoint detection using `convolutional neural network`. The goal is to predict facial landmarks positions.

<img src='exercise_09/images/key_pts_example.png'/>

## Handwritten Digits Classification on MNIST Using Autoencoder

MNIST dataset contains 60,000 images of handwritten digits. The first task is to train an autoencoder to reproduce these unlabeled images. Afterwards, the weights of the pre-trained encoder is transferred and fine-tuned on a classifier using the available labeled data.

![network split](/exercise_08/img/network_split.png)

### Classifier

- The encoder uses linear layers that map to latent space.
- The classifier is concatenated to the encoder, where it takes latent space input and classifies into digits classes. The classifier uses a block of fully connected layers.

## Image Classification on CIFAR-10 Using Neural Network

### Modularization

Using the chain rule, the model could be split into layers by using a `forward` and `backward` pass. The forward and backward passes are define according to the function of each layer as explained in the following sections.

### Sigmoid

- `forward`: given an input `X` of any shape, calculate the `Sigmoid` function for it and cache the input. Afterwards, return the `output` and `cache`.
- `backward`: given `upstream gradient dout` and `cache`, calculate `gradient w.r.t X`

### ReLU

- `forward`: given an input `X` of any shape, calculate the `ReLU` function for it and cache the output. Afterwards, return the `output` and `cache`.
- `backward`: given `upstream gradient dout` and `cache`, calculate `gradient w.r.t X`

### Affine Layers

- `forward`: given `x`, `w`, `b`, calculate the affine output and cache inputs. Return `output` and `cache`.
- `backward`: given `upstream gradient dout` and `cache`, calculate and return `dx`, `dw`, `db`.

### N-Layer Classification Network

- `forward`: given input data `X` containing `N` minibatches, calculate `affine forward`, `activation forward`. Return predicted value. Notice that last layer has only `affine forward` because the `activation forward` is included in the `cross-entropy/softmax`.
- `backward`: given `gradient w.r.t network output dy`, calculate `activation backward`, `affine backward`. Afterwards, return the `gradients w.r.t. model weights`

### Cross-Entropy/Softmax Loss from Logits

- `forward`: given `y_out logits`, `y_truth`, transform `logits` into a distribution using `softmax` while maintaining numerical stability. return `loss`.
- `backward`: given `y_out`, `y_truth`, return `cross-entropy loss gradients w.r.t. y_out`

### SGD + Momentum

- `update`: given `w`, `dw`, calculate new `v`, and updated wieghts `next_w`.

### Adam

- `update`: given `w`, `dw`, `config`, `lr`, return `next_w`, `config`.

![Classifier Teaser](./exercise_05/images/sgd-sgdm-adam.png)


## House Prices Classification using Logistic Regression

Categorizing houses into ```low-priced``` or ```expensive``` using simple logistic regression model. The data that we will use here is the HousingPrice dataset. Feeding some features in our classifier, the output should then be a score that determines in which category the considered house is.
![Classifier Teaser](./exercise_04/images/classifierTeaser.png)

### Binary Cross Entropy Loss

- forward pass: returns BCE loss
- backward pass: returns the gradient of the input to the loss function w.r.t to predicted y.

### Back-propagation

- sigmoid: activation function
- forward pass: returns predicted output, compute forward pass for each layer, save in cache for backward pass
- backward pass: returns the gradient of the weight matrix w.r.t. the upstream gradient

### Optimizer and Gradient Descent

- step: A vanilla gradient descent step. returns updated weight after one step

### Solver

- step: performs a forward pass, calculates the loss, backward pass, tells the optimizer to update the weights by 1 step

![Classifier Teaser](./exercise_04/images/train-val-loss.png)
![Classifier Teaser](./exercise_04/images/train-prediction.png)

## Dataset and Dataloader Classes

- Visualization
- Preprocessing
- Normalization
- Dataset Class: wrapper that loads data and performs preprocessing
- Dataloader Class: wrapper that loads data in batches
## References

[Python Tutorial](https://docs.python.org/3/tutorial/)

[NumPy Tutorial](http://cs231n.github.io/python-numpy-tutorial/)

[Chain Rule TUM](https://bit.ly/tum-article)

[Chain Rule Stanford](http://cs231n.stanford.edu/handouts/linear-backprop.pdf)

[One-Hot Encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)

[Logits](https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean/31045)

[Softmax](https://en.wikipedia.org/wiki/Softmax_function)

[Adam](https://ruder.io/optimizing-gradient-descent/)