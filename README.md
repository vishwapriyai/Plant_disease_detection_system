# Plant Disease Detection System

## Overview
This project implements a plant disease detection system using deep learning techniques. The model is trained to classify images of plant leaves into various disease categories. The application allows users to upload images and receive predictions on whether the plant is healthy or affected by a specific disease.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit Application](#streamlit-application)

## Features
- Image classification of plant diseases using a custom CNN model.
- Data augmentation techniques to improve model performance.
- Cross-validation for model reliability assessment.
- User-friendly web interface for disease detection using Streamlit.

## Technologies Used
- **Python**: The primary programming language used for implementation.
- **PyTorch**: For building and training the deep learning model.
- **Torchvision**: For image transformations and dataset handling.
- **Streamlit**: For creating the web application interface.
- **Pandas, NumPy, Matplotlib**: For data manipulation and visualization.

## Installation
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
    ```
## Usage
- Ensure that your environment is set up correctly with the necessary dependencies.
- Run the model training script:
```bash
python model.ipynb
```
- After training, run the Streamlit application:
```bash
streamlit run app.py
```
- Upload an image of a plant leaf to detect the disease.

## Model Training
The model is trained using a custom CNN architecture. The training process includes:
- Data augmentation techniques to improve model performance.
- Cross-validation to assess model reliability.
- Batch gradient descent for training the model.

## Streamlit Application
The Streamlit application allows users to upload images and receive predictions on plant diseases. The application uses the trained model to classify the uploaded images and display the results.
