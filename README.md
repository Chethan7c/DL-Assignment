# Deep Learning and Reinforcement Learning Assignment

---

## 1. Introduction

This assignment consists of a collection of **Deep Learning and Reinforcement Learning programs**, implemented using **Python, TensorFlow/Keras, NumPy, and other supporting libraries**.
The objective of the assignment is to understand, implement, and analyze core learning algorithms through practical experimentation.

All programs were executed successfully, and necessary corrections and improvements were applied to ensure correctness, robustness, and compatibility with modern libraries.

---

## 2. List of Programs Implemented

1. AlexNet Convolutional Neural Network
2. Cats vs Dogs Image Classification
3. Q-Learning Based Path Optimization
4. LSTM for Time Series Forecasting
5. Character-Level RNN Text Generation
6. Reinforcement Learning Based Tic-Tac-Toe Game

---

## 3. Program Descriptions

---

### 3.1 AlexNet.py — Convolutional Neural Network

**Modifications Made:**

* Updated optimizer usage to follow modern Keras standards
* Suppressed unnecessary warning and log messages for clean execution
* Added a safe execution block to allow standalone execution

---

### 3.2 CatDog.py — Cats vs Dogs Classification

**Modifications Made:**

* Removed hard-coded absolute file paths to improve portability
* Added dataset existence validation to prevent runtime errors
* Replaced deprecated optimizer parameters with supported alternatives

---

### 3.3 DeepReinforcementearning.py — Q-Learning

**Modifications Made:**

* Added reproducibility using a fixed random seed
* Modularized training logic for better readability and reuse
* Improved clarity and numerical stability of Q-value updates

---

### 3.4 LSTM.py — Time Series Forecasting

**Modifications Made:**

* Implemented portable dataset loading using relative paths
* Added validation split during training for better generalization
* Applied safe exception handling for optional model visualization

---

### 3.5 Rnn.py — Character-Level Text Generation

**Modifications Made:**

* Implemented temperature-based probabilistic sampling
* Improved randomness control during text generation for better output quality

---

### 3.6 TicTacToe.py — Reinforcement Learning Game

**Major Bug Fixed:**
Incorrect player symbol initialization was identified and corrected to ensure proper game logic.

**Additional Improvements:**

* Added main execution safety using `__main__` guard
* Reduced training rounds for faster execution without affecting learning
* Stabilized reward propagation logic during training

---

## 4. Software and Tools Used

* Python 3.9+
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 5. Execution Instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run programs:

```bash
python AlexNet.py
python CatDog.py
python DeepReinforcementearning.py
python LSTM.py
python Rnn.py
python TicTacToe.py
```


