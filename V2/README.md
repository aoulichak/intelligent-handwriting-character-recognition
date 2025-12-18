# Intelligent Character Recognition (ICR) - Version 2.2

This directory contains the advanced version of the ICR system (V2.2). Unlike Version 1, which focused on the initial pipeline, Version 2.2 implements **Fine-Tuning** and **Transfer Learning** strategies. It leverages a model pretrained on synthetic data and refines it using large-scale real-world datasets (A-Z Handwritten Alphabets and EMNIST) to achieve higher generalization and accuracy.

## Key Improvements in V2.2

* **Fine-Tuning Strategy**: Integrates real-world handwriting data to refine a pretrained CNN model.
* **Hybrid Dataset**: Combines the **Kaggle A-Z Handwritten Alphabets** dataset with **EMNIST Letters** for diverse character representation.
* **Enhanced Architecture**: The Convolutional Neural Network (CNN) now includes **Batch Normalization** after convolutional layers for faster convergence and **Dropout** layers to prevent overfitting.
* **Optimized Normalization**: Uses specific mean (0.7465) and standard deviation (0.2650) values calculated from the combined dataset for better input preprocessing.

## Project Structure

| File Name | Description |
| :--- | :--- |
| **ICR_V2_2.ipynb** | **Fine-Tuning & Training**<br>This Jupyter Notebook handles the advanced training pipeline. It loads the pretrained model, processes the combined A-Z/EMNIST datasets, applies the improved normalization, and executes the fine-tuning process. |
| **icr_gui_app_v2.2.py** | **V2.2 Interface**<br>The updated Graphical User Interface designed specifically for the V2.2 model. It includes the updated `ICR_V2_Model` class definition to ensure compatibility with the fine-tuned weights. |
| **icr_v2_best.pth** | **Prerequisite Model**<br>The notebook assumes the existence of a pretrained model (typically from V2 Phase 1) to start the fine-tuning process. |
| **icr_v2.2_best.pth** | **The Final Model**


## Prerequisites

In addition to the libraries used in V1 (PyTorch, PyQt5, NumPy, Pillow), this version specifically requires access to the following datasets for training:
1.  **A-Z Handwritten Alphabets** (CSV format)
2.  **EMNIST Letters** (Idx format)

## Usage Instructions

### 1. Training (Fine-Tuning)
To replicate the V2.2 results:
1.  Ensure you have the required datasets downloaded.
2.  Open **ICR_V2_2.ipynb**.
3.  Update the file paths in the notebook to point to your local dataset locations.
4.  Run the cells to perform data integration, orientation correction (for EMNIST), and fine-tuning.
5.  The notebook will save the final model weights (e.g., `icr_model_v2_2.pth`).

### 2. Running the V2.2 Application
To test the enhanced model:
1.  Ensure the trained model file is in the same directory.
2.  Run the application script:

```bash
python icr_gui_app_v2.2.py
```

The interface allows for drawing or uploading images. The underlying preprocessing now matches the specific normalization statistics of the V2.2 training set for accurate inference.

## Model Architecture (V2.2)

The `ICR_V2_Model` class implements a deeper and more robust architecture compared to V1:

* **Input**: 28x28 Grayscale images.
* **Feature Extraction**: Multiple Convolutional blocks.
* **Stabilization**: Batch Normalization applied after convolutions.
* **Regularization**: Dropout applied before the final Fully Connected layer.
* **Output**: Logits for 26 alphabetic classes (A-Z).

## Acknowledgments

* **Claude Sonnet 4.5**: Development assistance, architecture optimization, and documentation support provided by Claude Sonnet 4.5.
