# Introduction to Deep Learning
## Outline
- Dataset and Dataloader Classes
- Logistic Regression
- Neural Network and CIFAR-10 Classification

--- 

## Dataset and Dataloader Classes
- Visualization
- Preprocessing
- Normalization
- Dataloader

---

## Logistic Regression
Categorizing houses into ```low-priced``` or ```expensive``` using simple logistic regression model. The data that we will use here is the HousingPrice dataset. Feeding some features in our classifier, the output should then be a score that determines in which category the considered house is.
![Classifier Teaser](./exercise_04/images/classifierTeaser.png)

### Binary Cross Entropy Loss
- forward pass: returns BCE loss
- backward pass: returns the gradient of the input to the loss function w.r.t to predicted y.

### Backpropagation
- sigmoid: activation function
- forward pass: returns predicted output, compute forward pass for each layer, save in cache for backward pass
- backward pass: returns the gradient of the weight matrix w.r.t. the upstream gradient

### Optimizer and Gradient Descent
- step: A vanilla gradient descent step. returns updated weight after one step

### Solver
- step: performs a forward pass, calculates the loss, backward pass, tells the optimizer to update the weights by 1 step

![Classifier Teaser](./exercise_04/images/train-val-loss.png)
![Classifier Teaser](./exercise_04/images/train-prediction.png)

---

## Neural Network and CIFAR10 Classification
### Sigmoid

### ReLU

### Affine Layers

### N-Layer Classification Network

### Cross-Entropy/Softmax Loss from Logits

### Gradient Descent vs Stochastic Gradient Descent

### SGD + Momentum

### Adam

![Classifier Teaser](./exercise_05/images/sgd-sgdm-adam.png)
---

## References
[1] https://docs.python.org/3/tutorial/

[2] http://cs231n.github.io/python-numpy-tutorial/