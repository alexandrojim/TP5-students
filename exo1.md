# Discovering Physical Laws with Neurosymbolic AI

## Objectives
* Understand the limitations of "Black Box" Neural Networks in scientific contexts.
* Implement **Symbolic Regression (SR)** to discover analytical expressions from noisy data.
* Bridge the gap between machine learning output and physical constants (The "Neurosymbolic Bridge").

## Prerequisites
You will need a Python environment with the following libraries:
```bash
pip install numpy matplotlib scikit-learn pysr
```

## 1. The Scenario: The "Digital Galileo"
Imagine you are observing a simple pendulum in a lab. You vary the length of the string ($L$) and measure the time it takes for one full oscillation ($T$). Your goal is to find the mathematical law that relates $L$ to $T$.
Data Generation: Run the following code to simulate your experimental data. We add Gaussian noise to simulate real-world measurement errors.

```python
import numpy as np
import matplotlib.pyplot as plt

# Experimental Setup
g_true = 9.80665  # Earth gravity (m/s^2)
n_samples = 100
noise_intensity = 0.05 

# Independent variable: Length (m)
L = np.random.uniform(0.1, 2.5, n_samples)

# Dependent variable: Period (s) -> T = 2 * pi * sqrt(L/g)
T_clean = 2 * np.pi * np.sqrt(L / g_true)
T_noisy = T_clean + np.random.normal(0, noise_intensity, n_samples)

plt.scatter(L, T_noisy, alpha=0.6, label="Measurements")
plt.xlabel("String Length (L) in meters")
plt.ylabel("Period (T) in seconds")
plt.title("Pendulum Experiment Data")
plt.legend()
plt.show()
```

## 2. The Neural Baseline (Black Box)
Train a standard Multi-Layer Perceptron (MLP) to predict $T$ from $L$.
Task:
1. Train the model until the loss converges.
2. Observe the internal weights of the model.
3. Question: Can you find $g$ or $\pi$ inside these weights?

```python
from sklearn.neural_network import MLPRegressor

X = L.reshape(-1, 1)
y = T_noisy

mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=5000)
mlp.fit(X, y)

# Test extrapolation: Predicting for a 10m and 50m pendulum
L_extrap = np.array([[10.0], [50.0]])
preds = mlp.predict(L_extrap)
print(f"MLP Extrapolation for 10m: {preds[0]:.2f}s")
print(f"MLP Extrapolation for 50m: {preds[1]:.2f}s")
```

## 3. The Symbolic Discovery
Now we use PySR to search the space of mathematical expressions. This is the Symbolic part of the Neurosymbolic approach. It uses an evolutionary algorithm to find the simplest equation that fits the data.

Task: 
1. Run the regressor. 
2. Look at the output table. Identify the equation with the best score.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "*", "/", "^"],
    unary_operators=["sqrt"],
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
)

model.fit(X, y)
print(model.equations_)
```

## 4. The Neurosymbolic Bridge
Once the AI provides an equation like $T = 2.01 \cdot \sqrt{L}$:
1. Identify the Form: Match the AI output to the theoretical formula:
$$T = \frac{2\pi}{\sqrt{g}} \sqrt{L}$$
2. Extract the Constant: If your discovered constant is $C$, then $C = \frac{2\pi}{\sqrt{g}}$. Calculate the value of $g$ (Gravity).
3. Extrapolation Test: Use your discovered formula to predict the period for $L = 100m$. Compare this to the Neural Network's prediction. Which one follows the laws of physics?

# Lab Report Requirements
1. Visuals: Show the scatter plot of the data and the curve-fit of your discovered equation.
2. The Equation: Provide the LaTeX version of the best equation found by PySR.
3. Physics Discovery: Show your step-by-step calculation for $g$. How close did the AI get to $9.81 m/s^2$?
4. Discussion: Explain why the Neural Network failed at extrapolation ($L=50m$).Why is a symbolic model more useful for an engineer than a black-box model?

