# Intelligent Character Recognition (ICR)

A professional, modular project for Intelligent Handwriting Character Recognition (ICR) with three major versions, each demonstrating progressive improvements in model architecture, user interface, and deployment. This repository is designed for researchers, students, and developers interested in handwriting recognition using deep learning.

---

## Project Overview

This project provides three distinct versions of an ICR system:
- **V1:** Baseline CNN model with a simple GUI.
- **V2:** Enhanced models and improved GUI, with sub-versions for iterative improvements.
- **V3:** Advanced model (Keras), modernized GUI, and best performance.

Each version is self-contained and includes its own code, model weights, and documentation.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Version Details](#version-details)
- [Contributing](#contributing)
- [License](#license)

---

## Features

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pillow-8C8C8C?style=for-the-badge&logo=pillow&logoColor=white" alt="Pillow"/>
  <img src="https://img.shields.io/badge/PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt5"/>
</p>

- Handwriting character recognition using deep learning
- Multiple model architectures (PyTorch, Keras)
- User-friendly GUI applications for each version
- Modular and extensible codebase
- Ready-to-use pre-trained models

---

## Project Structure
```
v1/           # Version 1: Baseline CNN (PyTorch)
V2/           # Version 2: Improved models and GUI
  V2.1/       # Sub-version 2.1
  V2.2/       # Sub-version 2.2 (PyTorch)
V3/           # Version 3: Advanced model (Keras)
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aoulichak/intelligent-handwriting-character-recognition.git
   cd intelligent-handwriting-character-recognition
   ```
2. **Install dependencies:**
   - Each version contains its own `requirements.txt` (see inside `v1/` and `V3/`).
   - Install with pip:
     ```bash
     pip install -r v1/requirements.txt
     # or for V3
     pip install -r V3/requirements.txt
     ```

---

## Usage

- **V1:**
  ```bash
  python v1/icr_gui_app.py
  ```
- **V2.2:**
  ```bash
  python V2/V2.2/icr_gui_app_v2.2.py
  ```
- **V3:**
  ```bash
  python V3/icr_gui_app_v3.py
  ```

> For Jupyter notebooks, open the corresponding `.ipynb` file in your preferred environment.

---

## Version Details

### V1
- Baseline CNN model (PyTorch)
- Simple GUI for digit/character recognition
- Pre-trained weights: `icr_cnn_model.pth`

### V2
- Improved model architectures
- Enhanced GUI
- Sub-versions for iterative improvements:
  - **V2.1:** Documentation and minor updates
  - **V2.2:** Best PyTorch models (`icr_v2_best.pth`, `icr_v2.2_best.pth`)

### V3
- Advanced model (Keras, TensorFlow)
- Modern GUI
- Best accuracy and performance
- Pre-trained weights: `ICR_V3_final.keras`

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License.
See the LICENSE: (https://github.com/aoulichak/intelligent-handwriting-character-recognition/blob/main/LICENSE) file for details.

---

© Mohamed Aoulichak, 2025.
