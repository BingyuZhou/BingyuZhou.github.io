---
layout: post
comments: true
title:  "Understansing the forward and backward pass of Batch Normalization"
date:   2018-01-09 23:00:00
categories: main
---

Batch normalization, as it is proposed in [1], is a popular technique in deep learning to speed up the training progress and reduce the difficulty to train deep neural networks. As the authors in [1] hypothsize that the shifted distribution of the features may causes the training much harder, especially at deep layers. The deep learning methods usually works better when the input of each layer have uncorrelated feature with a zero mean and an unit variance. Therefore, researchers come up with ideas to build up batch normalization layers which try to normalize the input to a nice distribution with zero mean and unit variance. 

The batch normalization layer is usually before nonlinear layers, like *ReLU* and *tanh*. In order to be consistent with the minibatch optimization, the input features are normalized according to the chosen batch. And the *running mean* and *running variance* are updated during each batch by an update rate called *momentum*. Eventually, the *running mean* and *running variance* will be the estimated mean and variance for the whole training data set. However, not all layers prefer the normalized input during the training preocess. The smart idea of batch normalization is the design of two learning parameters $$\gamma$$ and $$\beta$$. They control how much to unnormalize the features. When $$\gamma=\sqrt{\sigma}$$ and $$\beta=\mu$$, the original unnormalized features are restored. By learning these two parameters, the preference of input's distribution for each batch normalization layer can be specified. After training, the learning parameters ($$\gamma, \beta$$) and the *running mean* and the *running variance* are determined. They can be used in the test mode to normalize the input.

## Forward pass

It is easy to understand and code the forward pass. Just pay attention that batch normalized is different in training and testing modes.

The forward pass can be described in the following formula:

$$
\begin{align*}
y &= \gamma \hat{x}+\beta\\
\hat{x} &= \frac{x-\mu}{\sqrt{\sigma^2}}\\
\mu &= \frac{\sum x}{n}\\
\sigma^2 &= \frac{\sum{(x-\mu)^2}}{n}
\end{align*}
$$

## Backward pass

It is always more difficult to understand and implement the backward pass in neural networks. It takes me some time to calculate the derative of the batch normalization layer. The main difficulty lies in the chain-rule of derative and the matrix's derivative. When you read in the following math calculation, please **keep the size of the matrix** in mind! After all, the change of the dimention is one beauty of tensors.

The backward pass is nothing else but to compute the deratives $$\frac{\partial f}{\partial x}$$, $$\frac{\partial f}{\partial \gamma}$$ and $$\frac{\partial f}{\partial \beta}$$.

First we need to write down the relationship between the intermediat varibles and its parameters:

$$
\begin{align*}
&y(\gamma, \beta, \hat{x})\\
&\hat{x}(x, \mu, \sigma^2)\\
&\mu(x)\\
&\sigma^2(\mu, x)
&\end{align*}
$$

Then it will be easier to compute the derivates $$\frac{\partial f}{\partial \gamma}$$ and $$\frac{\partial f}{\partial \beta}$$.

$$
\frac{\partial f}{\partial \gamma} = \sum{dout\hat{x}}
\frac{\partial f}{\partial \beta} = \sum{dout}
$$

As $$\hat{x}$$ is the normalized input, it is a matrix. But the $$\gamma$$ and $$\beta$$ are vectors. Therefore in consistent of the derative's size, we need to use $$\sum$$ along each feature (equivalent to each volume) to keep the size correct.

Next we will compute $$\frac{\partial f}{\partial x}$$.

I compute the derative in a manner of top-down.

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial \hat{x}}\frac{\partial \hat{x}}{\partial x} + \frac{\partial f}{\partial \mu}\frac{\partial \mu}{\partial x} + \frac{\partial f}{\partial \sigma^2}\frac{\partial \sigma^2}{\partial x}
$$

---

$$
\frac{\partial f}{\partial \hat{x}} = \frac{\partial f}{\partial y}\frac{\partial y}{\partial \hat{x}} = dout*\gamma
$$

---

$$
\frac{\partial \hat{x}}{\partial x} = np.ones(n)\sigma^{-1}
$$

where $$np.ones(n)$$ is a vector to shape the vector $\sigma$ into a matrix with the similar size of $$x$$.

---

$$
\frac{\partial f}{\partial \mu} = \sum{\frac{\partial f}{\partial \hat{x}}\frac{\partial \hat{x}}{\partial \mu}}+\frac{\partial f}{\partial \sigma^2}\frac{\partial \sigma^2}{\partial \mu}
$$

It is easy to calculate $$\frac{\partial f}{\partial \sigma^2}\frac{\partial \sigma^2}{\partial \mu}=0$$. And,

$$
\frac{\partial \hat{x}}{\partial \mu} = -\sigma^{-1}
$$

---

$$
\frac{\partial \mu}{\partial x} = np.ones(n, m) / n
$$

where (n, m) is the size of $$x$$.

---

$$
\begin{align*}
\frac{\partial f}{\partial \sigma^2} &= \sum{\frac{\partial f}{\partial \hat{x}}\frac{\partial \hat{x}}{\partial \sigma^2}} \\
\frac{\partial \hat{x}}{\partial \sigma^2} &= -\sum{(x-\mu)}\sigma^{-3}
\end{align*}
$$

---

$$
\frac{\partial \sigma^2}{\partial x} = \frac{2}{n}(x-\mu)
$$

---

Now, each component of $$\frac{\partial f}{\partial x}$$ is seperately computed. The final task is to use these components to calculate the derative:

$$
\frac{\partial f}{\partial x} = \frac{1}{\sigma n}(n\frac{\partial f}{\partial \hat{x}} - \sum\frac{\partial f}{\partial \hat{x}} - \hat{x}\sum{\frac{\partial f}{\partial \hat{x}}\hat{x}})
$$

In python, is should beauty

```python
var, gamma, beta, x_normal, bn_param = cache
eps = bn_param.get('eps', 1e-5)
dgamma = np.sum(dout*x_normal, axis=0)
dbeta = np.sum(dout, axis=0)
N = dout.shape[0]
dx = 1.0/(np.sqrt(var+eps))/N*(N*dout*gamma - np.sum(gamma*dout, axis=0) -
               x_normal*np.sum(gamma*dout*x_normal, axis=0))
```

Reference:

[1]: Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015.