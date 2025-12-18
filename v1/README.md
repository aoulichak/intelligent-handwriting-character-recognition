# Handwritten Character Recognition (ICR) System v1

This repository hosts an Intelligent Character Recognition (ICR) project designed to classify handwritten alphabets (A-Z). The project encompasses the complete pipeline, ranging from training a Deep Learning model to deploying a graphical user interface (GUI) for real-time inference.

## Project Structure

The repository consists of three main components, each serving a distinct role in the workflow:

| File Name | Description |
| :--- | :--- |
| **ICR_v1_jupyter.ipynb** | **Model Training (The Factory)**<br>This Jupyter Notebook contains the complete source code for the machine learning pipeline. It handles dataset loading, image preprocessing, CNN architecture definition, and the training loop. Executing this notebook generates the trained model file. |
| **icr_cnn_model.pth** | **Trained Weights (The Brain)**<br>This is the serialized PyTorch model file containing the learned weights and parameters. It is the direct output of the training notebook and serves as the necessary input for the GUI application. |
| **icr_gui_app.py** | **User Interface (The Application)**<br>A Python script that implements a Graphical User Interface (GUI). It loads the trained `.pth` model and provides an interactive canvas for users to draw or upload characters for real-time prediction. |

## Prerequisites

To run this project, ensure you have Python installed (version 3.8 or higher is recommended). The system relies on the following core libraries:

* **PyTorch**: For deep learning model inference.
* **PyQt5**: For the graphical user interface.
* **NumPy**: For numerical operations and array manipulation.
* **Pillow (PIL)**: For image processing.

You can install all necessary dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```
## Usage Instructions

### 1. Training the Model
To retrain the model or inspect the architecture and training process:

1. Open the `ICR_v1_jupyter.ipynb` file in Jupyter Notebook or Google Colab.
2. Execute all cells sequentially.
3. Upon successful execution, the notebook will export the trained model as `icr_cnn_model.pth`.

### 2. Running the Application
To launch the interface and test the model:

1. Ensure that `icr_cnn_model.pth` is located in the same directory as the script.
2. Execute the following command in your terminal:

```bash
python icr_gui_app.py
```
The application window will appear with the following features:
* **Drawing Canvas**: Use the mouse to draw a character on the black canvas.
* **Prediction**: Click the "Predict" button to classify the drawing.
* **Top-5 Results**: The interface displays the five most probable classes with their confidence scores.
* **Image Upload**: Users may upload external image files for classification.

## Model Architecture
The core of the system is a custom Convolutional Neural Network (CNN) implemented in PyTorch. The architecture includes:
* Three Convolutional Blocks (Conv2d, Batch Normalization, ReLU activation, and MaxPooling).
* Flattening and Fully Connected Layers for final classification.
* The model is trained on the A-Z Handwritten Alphabets dataset.

## Acknowledgments
* **Claude Sonnet 4.5**: Development assistance, code optimization, and documentation support provided by Claude Sonnet 4.5.
