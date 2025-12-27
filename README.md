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
   git clone https://github.com/yourusername/intelligent-handwriting-character-recognition.git
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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
