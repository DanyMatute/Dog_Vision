# 🐶 End-to-End MultiClass Dog Breed Classification
This notebook builds an end to end multiclass image classifier using TensorFlow 2.0 and Tensorflow Hub.

## 1. Problem
Identifing the breed of the dog given the image of a dog. There are 120 dogs.
## 2. Data
Data is from Kaggle.\
https://www.kaggle.com/competitions/dog-breed-identification/data
## 3. Evaluations
Submissions are evaluated on Multi Class Log Loss between the predicted probability and the observed target.
Submission File

For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```
## 4. Features
Information about the data:
- Unstructured data because they are images
- best if we use deep learning/transfer learning
- 120 different classes therefore multi-classification
- There are about 10,000 images in the training and 10,000 images in test set.
