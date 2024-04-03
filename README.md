
# **Multiclass Brain tumour Classification using Resnet-50**

This repository contains the code for a multiclass classification model trained to classify brain tumor images into four categories: pituitary tumor, meningioma tumor, glioma tumor, and no tumor. The model architecture used for this classification task is ResNet-50, a deep convolutional neural network known for its excellent performance in image classification tasks.

## **Problem Statement**

Brain tumor is the accumulation or mass growth of abnormal cells in the brain. There are basically two types of tumors, malignant and benign. Malignant tumors can be life-threatening based on the location and rate of growth. Hence timely intervention and accurate detection is of paramount importance when it comes to brain tumors. This project focusses on classifying 3 types of brain tumors based on its loaction from normal cases i.e no tumor using Convolutional Neural Network.

## **Dataset**
The dataset used for this model is taken from Brain Tumor MRI Dataset available on Kaggle. The distribution of images in training data are as follows:
- Pituitary tumor (916)
- Meningioma tumor (906)
- Glioma tumor (900)
- No tumor (919) 

The distribution of images in testing data are as follows:
- Pituitary tumor (200)
- Meningioma tumor (206)
- Glioma tumor (209)
- No tumor (278) 

## **Image Preprocessing**
Image preprocessing is applied to all the images in the dataset
1. Cropping the image : removes the unwanted background noise. Thus helping the algorithm to focus completely on the features of interest

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/5ce30227-f438-4fc2-bd25-8ee51f0c828b" alt="Description of the image">
</p>
<p align="center">
  Images after Cropping
</p>

2.	Noise Removal : Bilateral filter is used for noise removal. It smooths the image while preserving edges and fine details. Bilateral filter considers both the spatial distance and intensity similarity between pixels when smoothing the image. Hence suitable for processing MRI images acquired with different imaging protocols and parameters.
3.	Applying colormap : Applying a colormap can improve the interpretability of MRI images by enhancing the contrast between different tissues or structures
4.	Resize : Resizing the image for standardizing the input size of images to be fed into a machine learning model

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/0fc6573c-2c9e-43b6-afcc-cd61a2c7172c" alt="Description of the image">
</p>
<p align="center">
  Images after preprocessing
</p>


## **Splitting the data into train, test and validation**
Here the train data is split into train and validation sets. The test data is completely unseen. There are 2912 train images, 729 validation images an 893 test images.


## **Image Augmentation using Image Data Generator**
Medical imaging datasets, including MRI images, are often limited in size due to factors such as data collection constraints, privacy concerns, or rarity of certain conditions. Image augmentation allows to artificially increase the size of the dataset by generating variations of existing images. Augmentation can help prevent the model from memorizing specific patterns or features in the training data that may not generalize well to unseen data, thus leading to a more robust and generalizable model.

## **Model Training**

Resnet-50 is used for training the brain tumor dataset. ResNet-50’s increased depth allows it to capture more intricate patterns and features in the data, which can be beneficial for detecting complex structures in brain tumor images. By transfer learning, ResNet-50’s pre-trained weights from ImageNet are leveraged to bootstrap training on the brain tumor classification task. 

## **Results**
The following results have been achieved with Resnet-50 model for detection of Glioma, Meningioma, Pituitary and Normal patients from Brain MRI images.

- Test Accuracy      : 97%
- f1-score (glioma)  : 97%
- f1-score (meningioma) : 96%
- f1-score (pituitary) : 96%
- f1-score (no_tumorl) : 100%


**Confusion matrix**

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/98b38811-b4ef-4ad6-b3a1-d6a75b25219a">
</p>

**Sample predictions**

Predicted label(True label)

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/9dc134eb-2753-46ea-b081-2a9793c55e3d">
</p>

## **Streamlit App**

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/fe226ebe-d080-420b-83b7-c81e7ed37df7" alt="Description of the image">
</p>
<p align="center">
  Prediction for Pituitary tumor
</p>

## **Future work**
- Include more image preprocessing steps so as to extract intricate details correctly 
- Increase the number of samples in the dataset
