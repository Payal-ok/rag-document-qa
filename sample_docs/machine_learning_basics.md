# Machine Learning Basics

Machine learning is a way of building programs that learn patterns from data instead of following hand written rules. A model is trained on examples and is then used to make predictions about new data.

## Types of learning

In supervised learning, every training example has a label, and the model learns to predict that label. Predicting a house price is a regression task, while deciding whether an email is spam is a classification task. In unsupervised learning there are no labels. Clustering, such as k-means, groups similar points together, and PCA reduces the number of features while keeping most of the variance.

## Overfitting

A model overfits when it memorises the training data, noise included, and then performs badly on new data. To detect overfitting, the data is split into a training set and a test set. A large gap between training accuracy and test accuracy is a warning sign. Remedies include collecting more data, using a simpler model, and adding regularisation, which penalises large weights.

## Gradient descent

Gradient descent is an algorithm that trains many models. It computes how the loss changes with each weight and moves the weights a small step in the direction that lowers the loss. The size of the step is the learning rate. If the learning rate is too large, training can diverge. If it is too small, training becomes very slow.
