# Introduction to Deep Learning
## Outline
 - ### Data
 - ### Loss
 - ### Neural Networks
 - ### References

--- 

## Data
- Dataloader
- Visualization
- Preprocessing
- Normalization

---

## Loss
### Binary Cross Entropy Loss
- forward pass: returns BCE loss
- backward pass: returns the gradient of the input to the loss function w.r.t to predicted y.

---

## Neural Networks
### Backpropagation
- sigmoid: activation function
- forward pass: returns predicted output, compute forward pass for each layer, save in cache for backward pass
- backward pass: returns the gradient of the weight matrix w.r.t. the upstream gradient

### Optimizer and Gradient Descent
- step: A vanilla gradient descent step. returns updated weight after one step

### Solver
- step: performs a forward pass, calculates the loss, backward pass, tells the optimizer to update the weights by 1 step

---

## References
[1] https://docs.python.org/3/tutorial/

[2] http://cs231n.github.io/python-numpy-tutorial/