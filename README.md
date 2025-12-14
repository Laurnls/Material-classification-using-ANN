# Material Classification Using Artificial Neural Network (ANN)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

> **Klasifikasi Material Konduktor dan Isolator menggunakan Artificial Neural Network (ANN)**  
> Computational Physics II Project - Physics Department, Universitas Airlangga

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)
- [Citation](#citation)

---

## 🎯 Overview

This project implements a **simple Artificial Neural Network (ANN)** for classifying materials into two categories: **Conductors** and **Insulators** based on their physical properties (electrical resistivity and thermal conductivity).

### Key Highlights:
- ✅ **100% accuracy** on test data
- ✅ **96.67% cross-validation accuracy** (5-fold)
- ✅ Comprehensive hyperparameter sensitivity analysis
- ✅ Feature importance analysis using permutation method
- ✅ Interactive visualizations (9-panel comprehensive plot)

### Physics Motivation:
Materials exhibit drastically different electrical properties spanning over **22 orders of magnitude** in resistivity (from 10⁻⁸ Ω·m for silver to 10¹⁴ Ω·m for glass). This project demonstrates how machine learning can recognize these patterns without explicit mathematical modeling.

---

## ✨ Features

- **Data Preprocessing:**
  - Logarithmic transformation of resistivity values
  - Z-score normalization using StandardScaler
  - Stratified train-test split (70:30)

- **Model Architecture:**
  - Multi-layer Perceptron (MLP) with optimized hidden layers
  - ReLU activation function
  - Adam optimizer

- **Comprehensive Analysis:**
  - Hyperparameter sensitivity (hidden layers, activation functions)
  - 5-fold cross-validation
  - Confusion matrix and classification metrics
  - Feature importance analysis
  - Decision boundary visualization

- **Visualization:**
  - 9-panel comprehensive analysis plot
  - Original vs normalized data distribution
  - Decision boundary heatmap
  - Performance metrics comparison

---

## 📊 Dataset

### Material Properties:

**15 Conductor Samples:**
- Silver, Copper, Gold, Aluminum, Tungsten
- Iron, Platinum, Tin, Nickel, Zinc
- Brass, Carbon Steel, Lead, Magnesium, Titanium

**15 Insulator Samples:**
- Glass, Pyrex Glass, Quartz, Ceramic, Porcelain
- Teflon, Polyethylene, PVC, Rubber, Wood
- Paper, Mica, Epoxy Resin, Silicon, Bakelite

### Features:
1. **x₁:** log₁₀(Resistivity) in Ω·m
2. **x₂:** Thermal Conductivity in W/(m·K)

### Labels:
- **0:** Insulator
- **1:** Conductor

---

## 🚀 Installation

### Prerequisites:
- Python 3.8 or higher
- pip package manager

### Clone Repository:
```bash
git clone https://github.com/yourusername/ann-material-classification.git
cd ann-material-classification
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start:
```bash
python material_classification.py
```

---

## 💻 Usage

### Basic Usage:

```python
from material_classifier import MaterialClassifier

# Initialize classifier
classifier = MaterialClassifier()

# Load and preprocess data
classifier.load_data()
classifier.preprocess()

# Train model
classifier.train()

# Evaluate
results = classifier.evaluate()
print(f"Test Accuracy: {results['accuracy']*100:.2f}%")

# Visualize
classifier.visualize_results()
```

### Predict New Materials:

```python
# Define new material properties
new_materials = [
    [-7.5, 350],   # High conductivity, low resistivity → Conductor
    [12.0, 0.5],   # Low conductivity, high resistivity → Insulator
]

# Predict
predictions = classifier.predict(new_materials)
print(predictions)
```

### Custom Hyperparameters:

```python
classifier = MaterialClassifier(
    hidden_layers=(15, 10),
    activation='relu',
    learning_rate=0.001,
    max_iter=2000
)
```

---

## 📈 Results

### Model Performance:

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **Accuracy** | 100.00% | 100.00% |
| **Precision** | - | 100.00% |
| **Recall** | - | 100.00% |
| **F1-Score** | - | 100.00% |

### Cross-Validation (5-Fold):
- **Mean Accuracy:** 96.67%
- **Standard Deviation:** 6.67%
- **Fold Scores:** 100%, 83.3%, 100%, 100%, 100%

### Confusion Matrix (Test Data):
```
                Predicted
              Insulator  Conductor
Actual    
Insulator        5          0
Conductor        0          4
```

### Feature Importance:
- **log₁₀(Resistivity):** 0.4444 (dominant)
- **Thermal Conductivity:** ~0.0000 (negligible)

### Visualization:

![Comprehensive Analysis](images/comprehensive_analysis.png)

*Nine-panel analysis showing: (a) original data distribution, (b) normalized data, (c) decision boundary, (d) confusion matrix, (e) accuracy comparison, (f) hidden layer sensitivity, (g) activation function comparison, (h) feature importance, (i) cross-validation results.*

---

## 📁 Project Structure

```
ann-material-classification/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore file
│
├── data/
│   ├── material_data.csv             # Dataset
│   └── data_description.md           # Data documentation
│
├── src/
│   ├── __init__.py
│   ├── material_classifier.py        # Main classifier class
│   ├── data_loader.py                # Data loading utilities
│   ├── preprocessing.py              # Preprocessing functions
│   ├── model.py                      # ANN model definition
│   ├── evaluation.py                 # Evaluation metrics
│   └── visualization.py              # Plotting functions
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb # EDA notebook
│   ├── 02_model_training.ipynb       # Training notebook
│   └── 03_results_visualization.ipynb # Results notebook
│
├── scripts/
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   └── predict.py                    # Prediction script
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_evaluation.py
│
├── docs/
│   ├── paper/
│   │   ├── main.tex                  # LaTeX journal paper
│   │   ├── main.pdf                  # Compiled PDF
│   │   └── references.bib            # Bibliography
│   │
│   ├── methodology.md                # Detailed methodology
│   ├── results.md                    # Detailed results
│   └── api_reference.md              # API documentation
│
├── images/
│   ├── comprehensive_analysis.png    # Main results figure
│   ├── architecture.png              # Model architecture
│   └── workflow.png                  # Project workflow
│
├── models/
│   ├── best_model.pkl                # Trained model
│   └── scaler.pkl                    # Fitted scaler
│
└── examples/
    ├── basic_usage.py                # Basic example
    ├── custom_hyperparameters.py     # Custom config example
    └── batch_prediction.py           # Batch prediction example
```

---

## 📚 Documentation

### Methodology:
Detailed methodology is available in [`docs/methodology.md`](docs/methodology.md)

### API Reference:
Complete API documentation: [`docs/api_reference.md`](docs/api_reference.md)

### Academic Paper:
LaTeX source and compiled PDF available in [`docs/paper/`](docs/paper/)

### Jupyter Notebooks:
Interactive tutorials in [`notebooks/`](notebooks/)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup:
```bash
# Clone your fork
git clone https://github.com/yourusername/ann-material-classification.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

### Code Style:
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Add unit tests for new features

---

## 👥 Authors

**Group 7 - Computational Physics II**  
Physics Department, Faculty of Science and Technology  
Universitas Airlangga, 2025

- **Laurina Salsabilah** - 082111333052
- **Luk Luil Maknun** - 182231004
- **Ahmad Haris Saiful Anam** - 182231040
- **Novita Aulia Rafi** - 182231052

**Supervisor:**  
Dr. Ir. Soegianto Soelistiono, M.Si.  
NIP. 197001251993031003

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Group 7 - Computational Physics II

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@article{salsabilah2025ann,
  title={Classification of Physical Material Data Using Simple Artificial Neural Network Based on Scikit-learn},
  author={Salsabilah, Laurina and Maknun, Luk Luil and Anam, Ahmad Haris Saiful and Rafi, Novita Aulia},
  journal={Physics Department, Universitas Airlangga},
  year={2025}
}
```

**APA Format:**
```
Salsabilah, L., Maknun, L. L., Anam, A. H. S., & Rafi, N. A. (2025). 
Classification of Physical Material Data Using Simple Artificial Neural 
Network Based on Scikit-learn. Physics Department, Universitas Airlangga.
```

---

## 🙏 Acknowledgments

- **Scikit-learn** for the excellent machine learning library
- **Matplotlib** and **Seaborn** for visualization tools
- **NumPy** for numerical computations
- Dr. Ir. Soegianto Soelistiono, M.Si. for guidance and supervision
- Physics Department, Universitas Airlangga for research facilities

---

## 📞 Contact

For questions or collaboration:

- **Email:** laurina.salsabilah@example.com
- **GitHub Issues:** [Create an issue](https://github.com/yourusername/ann-material-classification/issues)
- **Discussion:** [GitHub Discussions](https://github.com/yourusername/ann-material-classification/discussions)

---

## 🔗 Related Projects

- [Deep Learning for Materials Science](https://github.com/example/dl-materials)
- [Physics-Informed Neural Networks](https://github.com/example/physics-nn)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)

---

## 📊 Project Status

**Current Version:** 1.0.0  
**Status:** ✅ Active Development  
**Last Updated:** December 2025

### Roadmap:
- [ ] Add support for multi-class classification (semiconductors)
- [ ] Implement deep learning models (PyTorch/TensorFlow)
- [ ] Create web interface for predictions
- [ ] Add more material properties as features
- [ ] Develop ensemble methods
- [ ] Create Docker container for easy deployment

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/ann-material-classification&type=Date)](https://star-history.com/#yourusername/ann-material-classification&Date)

---

<div align="center">

**Made with ❤️ by Group 7 Computational Physics II**

[⬆ Back to Top](#material-classification-using-artificial-neural-network-ann)

</div> 
