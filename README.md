# CNN Analysis: Optimization, Invariance, and Vulnerability

This repository contains a comprehensive 4-part project analyzing the behavior, properties, and vulnerabilities of Convolutional Neural Networks (CNNs). The experiments move from building and optimizing a baseline model to advanced robustness testing.

The project uses models like AlexNet and ResNet-18 on datasets such as CIFAR-10 and the Oxford-IIIT Pet dataset. The key Python libraries used are `PyTorch`, `Optuna` for hyperparameter tuning, and `Scikit-learn` for classifier analysis.

**Code Availability:** The complete code for all four experiments described in this report is contained in the Jupyter Notebooks in this repository.

## Table of Contents
* [Project Overview](#project-overview)
* [Experiments](#experiments)
* [Key Results & Visualizations](#key-results--visualizations)
* [Setup and Installation](#setup-and-installation)
* [Usage](#usage)
* [License](#license)

## Project Overview

This project is structured as a series of four experiments, each building on the last, to provide a complete analysis of CNNs.

1.  **Optimization:** We first build a strong baseline model by systematically tuning both the architecture and hyperparameters of a custom AlexNet on CIFAR-10 using Optuna.
2.  **Transfer Learning:** We then explore feature extraction, using a pre-trained AlexNet on the Oxford-IIIT Pet dataset to show how "deep features" can be used to train simple, classical machine learning models.
3.  **Invariance:** We investigate the inherent properties of CNNs by testing our model's robustness to geometric transformations (translation, rotation, and flipping).
4.  **Vulnerability:** Finally, we demonstrate a critical security flaw by performing adversarial attacks (untargeted FGSM and targeted PGD) to fool both our custom model and a state-of-the-art ResNet-18.

## Experiments

This repository contains the Jupyter Notebook for each experiment:

1.  **Experiment 1: Hyperparameter Tuning (Baseline Model)**
    * **Notebook:** `AlexNetHyperparameterTunning.ipynb`
    * **Description:** Builds a custom AlexNet-style CNN for CIFAR-10 and uses `Optuna` with 3-Fold Cross-Validation to find the optimal architecture and training parameters. The final optimized model achieves **87.38%** test accuracy.

2.  **Experiment 2: Transfer Learning (Feature Extraction)**
    * **Notebook:** `featureExtract.ipynb`
    * **Description:** Uses a pre-trained AlexNet on the 37-class Oxford-IIIT Pet dataset as a fixed feature extractor. The 4096-dimensional output vectors are then used to train `LogisticRegression` (74.5%) and `RandomForestClassifier` (72.4%) models.

3.  **Experiment 3: CNN Invariance Testing**
    * **Notebook:** `CNNInvariance.ipynb`
    * **Description:** Tests the baseline model from Experiment 2 against various geometric transformations.
    * **Finding:** The model is highly invariant to translation and horizontal flipping but **not** invariant to rotation, demonstrating a key limitation of standard CNNs.

4.  **Experiment 4: Adversarial Attacks**
    * **Notebook:** `CNN_Adverserial_ML.ipynb`
    * **Description:** Implements two types of adversarial attacks:
        * **Untargeted FGSM:** Successfully tricks the CIFAR-10 AlexNet into misclassifying a 'cat' as a 'dog' with high confidence.
        * **Targeted PGD:** Successfully tricks a pre-trained ResNet-18 into misclassifying a 'dog' as an 'ostrich' with high confidence.

## Key Results & Visualizations

### 1. Untargeted FGSM Attack on CIFAR-10 (Exp 4)
This image shows a successful untargeted FGSM attack. The original image (left), correctly identified as a 'cat' (80.4% confidence), has imperceptible noise (center) added. The resulting adversarial image (right) is confidently misclassified as a 'dog' (85.4% confidence).

![FGSM Attack on CIFAR-10](image_4b033a.png)

### 2. Targeted PGD Attack on ImageNet (Exp 4)
This plot shows a powerful targeted PGD attack. The model's original Top-5 prediction (left) is 'Samoyed' with ~89% confidence. The attack, targeting 'ostrich', completely flips the model's output, which now predicts 'ostrich' (right) with ~89% confidence, while the original class vanishes from the top 5.

![PGD Attack on ImageNet](image_fd039d.png)

## Setup and Installation

This project was built using Python 3. The primary requirements can be installed via pip.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/cnn-robustness-analysis.git](https://github.com/YOUR_USERNAME/cnn-robustness-analysis.git)
    cd cnn-robustness-analysis
    ```

2.  **Install requirements:**
    It is recommended to use a virtual environment.
    ```bash
    pip install torch torchvision
    pip install optuna numpy pandas
    pip install matplotlib seaborn scikit-learn
    pip install jupyterlab
    ```
    Alternatively, you can create a `requirements.txt` file and run `pip install -r requirements.txt`.

## Usage

1.  Start the Jupyter environment:
    ```bash
    jupyter lab
    ```
    or
    ```bash
    jupyter notebook
    ```
2.  Open the notebooks in numerical order, starting with `AlexNetHyperparameterTunning.ipynb`, to follow the experimental procedure.
3.  The notebooks are self-contained and include all code for data loading, model training, and analysis.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
