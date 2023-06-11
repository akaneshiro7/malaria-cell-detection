# Malaria Detection using FastAI

This repository contains a machine learning model built with the FastAI library, designed to detect whether a cell is infected with malaria. The model uses a Convolutional Neural Network (CNN) with a ResNet34 architecture, trained on a dataset of 27,558 cell images, sourced from the official NIH website [Malaria Dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/).

## Getting Started

To run the code, clone the repository to your local machine or cloud environment on Google Colab, Jupyter Notebooks, or Kaggle.

### Project Structure
The main script `malaria_detection.py` includes all steps in the data preprocessing, model training, and model evaluation. 

## Methodology

1. **Data Import and Preprocessing**: The cell images are located in the directory specified by the 'path' variable. The 'dls' Datablock splits the data into training and validation sets with a 80-20 split, applies resizing and aug_transforms.

2. **Model Training**: The model is a FastAI CNN learner with the resnet34 architecture. The learning rate is set by plotting loss against learning rate and selecting the learning rate corresponding to the steepest negative slope. The model is fine-tuned over 8 epochs.

### Loss vs. Learning Rate Graph for Optimal Learning Rate
![](https://github.com/akaneshiro7/malaria-cell-detection/blob/main/Loss%20vs%20Learning%20Rate.png)

3. **Model Evaluation**: The model performance is assessed by plotting a confusion matrix and the top losses. Grad-CAM heatmaps are also generated to visualize which parts of the image most influenced the model's predictions.
### Graph of Original Image next to GradCAM Heat Map
![](https://github.com/akaneshiro7/malaria-cell-detection/blob/main/malaria.png)
## Results
The model shows impressive 98.64% accuracy in distinguishing between parasitized and uninfected cells. The confusion matrix, top losses, and Grad-CAM heatmaps provide further insight into the model's performance and decision-making process.

### Model Accuracy when Trained over 8 epochs
![](https://github.com/akaneshiro7/malaria-cell-detection/blob/main/epoch.png)

### 16 Random Images and Predictions from Validation Set
![](https://github.com/akaneshiro7/malaria-cell-detection/blob/main/results.png)

## Acknowledgements
The dataset used in this project is sourced from the official NIH website [Malaria Dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/).
