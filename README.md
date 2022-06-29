# x-ray-pneumonia-detection

Implementation of a CNN for classifying the presence of pneumonia in X-ray images, with the goal of assisting radiologists. Details on the architecture, dataset, and training process are given below, and a mock FDA submission for the trained model can be found in FDA_submission.pdf.

The classification algorithm takes in DICOM files, checks the metadata for correct patient position, body part, and modality, and resizes the image into 224x224 pixels with 3 rgb colour channels. These resized images are classified with a deep CNN with a single logit output. If the output value is above a pre-calibrated threshold, the image is classified as Pneumonia, else No Pneumonia. See below for a schematic.

<img src="https://github.com/callumcanavan/x-ray-pneumonia-detection/blob/main/images/architecture.png" alt="drawing" width="900"/>

Data used for training and validating the model were obtained from the NIH Chest X-ray Dataset [1](https://arxiv.org/abs/1705.02315), wherein 14 disease labels (including pneumonia) specifying the presence of each disease were created using natural language processing to mine diagnoses from the radiological report associated with each image.

In this project, the training dataset consisted of 2310 images with a 1:1 ratio of positive to negative pneumonia cases. The gender distribution was slightly skewed towards males and patient ages were mostly within the range 20-70. The validation dataset consisted of 1104 images with a 1:3 ratio of positive to negative pneumonia cases. The gender distribution was skewed towards males (more so than in the training data) and patient ages were mostly within the range 20-70.

The classifying model consists of a deep convolutional neural network (VGG16) comprised of 13 convolution layers with ReLU activation functions and 5 max pooling layers, followed by 3 dense hidden layers with ReLU activation functions and a final dense output layer with one sigmoid-activated neuron (see above diagram for more details). This architecture is used because it can identify complex patterns of varying sizes within an image, some of which may indicate the presence or absence of pneumonia. 

Sample images used for model training were resized into 224x224 images with 3 rgb colour channels (for compatibility with the VGG16 convolutional network structure) and samplewise scaled (to prevent vanishing/exploding gradients). For data augmentation during training they were also randomly horizontally flipped (probability 0.5), heightshifted, width-shifted, rotated, sheared and zoomed. The scales for height-shifting, width shifting, shearing and zooming were each chosen from a uniform distribution in the range [0, 0.1] and the angle of rotation in degrees was chosen from a uniform distribution in the range [-10, 10] for each image in each epoch. The images were forward-propagated through the model in batches of 64, chosen for a tradeoff between training speed and computational efficiency. The batch size for validating the model after each epoch and testing the model after training was chosen to be 32 since training speed was not a factor here. Adam (with default β1= 0.9 and β2 = 0.999) was chosen as the optimizing algorithm during model training due to its combination of momentum and RMSProp properties making it a historically good default for complex image classification problems. A learning rate of 1e-4 was used after trying several rates in the [1e-3, 1e-5] log range and finding this learning rate to give fastest convergence (within around 12 epochs) and lowest loss on the validation set. The pretrained convolutional layers of this model were taken from VGG16, a model originally trained on ImageNet to classify images across 1000 categories. All convolutional layers except the last one had their weights frozen during training, meaning the last convolutional layer was the only one which was fine-tuned. The dense layers at the end of VGG16 were removed and replaced with 4 new dense layers of sizes 1024, 512, 256 and 1, with the last layer representing the output. The first 3 dense layers are relu activated while the last is sigmoid activated. These were subsequently trained on the training set with random dropout (rate 0.4) after each hidden layer for regularization. The training loss (binary cross entropy of classifications, also known as log loss) decreased steadily over all training epochs. The validation loss was also found to decrease but did so more erratically, and seemed to converge by the 15th epoch (training the model for more epochs after this led to overfitting).

<img src="https://github.com/callumcanavan/x-ray-pneumonia-detection/blob/main/images/train.png" alt="drawing" width="450"/>

<img src="https://github.com/callumcanavan/x-ray-pneumonia-detection/blob/main/images/pr.png" alt="drawing" width="450"/>

The above figure shows the tradeoff between precision and recall on the test set. The threshold (0.562) between positive and negative cases given the final output of the model was chosen to optimise algorithm’s F1 score (0.513), the harmonic mean of precision and recall. This gave a precision of 0.460 and recall of 0.560 on the test set.

[1] [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, Wang et al., 2017](https://arxiv.org/abs/1705.02315)

I completed this project as part of the Udacity AI for Healthcare nanodegree which provided blank notebooks and several other functionalities. Algorithm implementations, parameter tuning,  writeup/interpretation of results, and the mock FDA submission were completed by myself.

# Depends
```
keras scikit-learn numpy
```
