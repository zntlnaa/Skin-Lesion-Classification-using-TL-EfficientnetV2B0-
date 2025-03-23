# Skin Lesion Multiclass Classification Using Transfer Learning with EfficientNetV2B0 Architecture (My Final Project)

This study implements transfer learning using the EfficientNetV2B0 architecture for multiclass skin lesion classification.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Tools & Libraries](#tools-&-libraries)

## Project Overview
Skin lesion classification is crucial due to the high incidence of skin cancer, with over 1.5 million new cases reported in 2022. The ambiguous characteristics of some lesions often cause hesitation in seeking medical attention, while limited resources and trained medical personnel can delay diagnosis and treatment. Therefore, an automated computer-based classification method is needed. This study implements transfer learning by modifying the top layers of the EfficientNetV2B0 architecture, adding dense layers, dropout layers, and a dense output layer. This research aims to contribute to the development of a deep learning-based skin lesion classification model, particularly using transfer learning with EfficientNetV2B0.

## Data Sources
- **ISIC 2019 Dataset**: This study uses the ISIC 2019 dataset, which includes 4,522 images of Melanoma, 12,875 images of Melanocytic Nevi, 3,323 images of Basal Cell Carcinoma, and 2,624 images of Benign Keratosis.

## Methodology
### 1. Data Collection
This study uses the ISIC 2019 dataset. Four classes were selected from this dataset: 4,522 images of Melanoma, 12,875 images of Melanocytic Nevi, 3,323 images of Basal Cell Carcinoma, and 2,624 images of Benign Keratosis.

### 2. Data Splitting
The dataset was split into three parts: 70% training data, 10% validation data, and 20% test data.

### 3. Pre-processing
- **Data Augmentation**: Augmentation was only applied to the training data. The augmentation involved geometric transformations, including rotations by 90°, 180°, and 270°, and horizontal flipping. This technique was applied to the minority classes in the training data to balance the distribution to match the majority class, reaching 9,012 samples. Keras' ImageDataGenerator was used to generate additional variations dynamically during the training process.
- **Image Resizing**: All images were resized to 224 × 224 pixels, as required by the EfficientNetV2B0 model to ensure consistent image dimensions.
- **Pixel Normalization**: Pixel values were normalized using the rescaling and normalization layers built into the EfficientNetV2B0 model.

### 4. Model Building
- The model was built by modifying the top layers of the EfficientNetV2B0 model, adding a dense layer (512 neurons, ReLU activation), a dropout layer, and a dense output layer (4 neurons, Softmax activation). All layers were fine-tuned to improve the model's performance in classifying skin lesions.

### 5. Model Training
- The model was trained for 20 epochs with a batch size of 20, using the ReduceLROnPlateau technique to adjust the learning rate.

### 6. Model Testing
- The model was tested using the pre-prepared test data.

### 7. Model Evaluation
- The model was evaluated using three metrics: test accuracy, macro average recall, and macro average precision.

## Results
The model achieved a test accuracy of 88.66%, a macro average recall of 86.54%, and a macro average precision of 86.70%.

## Conclusion
For future research, it is recommended to further explore data augmentation techniques to optimize model performance and to include additional classes for more specific skin lesion classification. This study is limited to the EfficientNetV2B0 architecture, so a comparative evaluation with other deep learning architectures is needed to explore the potential for further performance improvements.

## Tools & Libraries
- Jupyter Notebook
- Numpy
- OpenCV
- Matplotlib
- Keras
- TensorFlow
