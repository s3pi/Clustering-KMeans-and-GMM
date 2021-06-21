### Clustering of Phonemes and Segmentation of Image data

In the first part of the assignment, we assume that data from a single class, is coming from multi-modal
Gaussian distribution. But we do not know how many modes / clusters are there in the data. We
begin by considering the problem of identifying groups or clusters of data points in a
multidimensional space. So, we perform the experiment on different number of modes (k’s) and
compute the accuracy in each case.
After assuming a certain k, we neither know, the membership of point belongs to which cluster
and the mean of each cluster. This is a chicken and egg problem. To solve this, K-Means
clustering technique is used.

In the second part of the assignment, we apply the concept of GMM to build the model for the given nonlinearly
separable data. We study the performance of the classifier by varying only the number of mixture
components (K)

In conclusion, Non linear Separable data’s accuracy is higher when we considered that the data from a single
class is coming from several Gaussian Mixture models.
Real World data also shows similar behaviour.
The decision boundary obtained in Non Linear and Real world data is very sophisticated hyper
plane compared to the decision boundary obtained with Bayesian Classifier.

Scene Image data, since we considered non overlapping 32*32 patches, the accuracy generally
low. 

Cell Image data, since we considered overlapping patches, although we ran GMM with reduced
set of training data points, segmentation appears close to the original image.
