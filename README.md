# Regression Using TensorFlow

Predicting fuel efficiency (MPG) from car specifications using a deep neural network built with TensorFlow/Keras.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wsamuelw/regression-using-tensorflow/blob/main/regression_using_tensorflow.ipynb)

## Problem

The Auto MPG dataset contains specifications for ~400 cars (cylinders, displacement, horsepower, weight, acceleration, model year, origin) and their fuel efficiency in miles per gallon. The goal: build a neural network that predicts MPG from these features.

## Approach

1. **Clean** — impute 6 missing Horsepower values with the mean
2. **Encode** — convert Origin (1/2/3) to one-hot vectors (USA/Europe/Japan)
3. **Scale** — normalise features and labels to zero mean, unit variance with `StandardScaler`
4. **Build** — 3-layer sequential network: Dense(32, relu) → Dense(64, relu) → Dense(1)
5. **Train** — 100 epochs, RMSProp optimiser, MSE loss
6. **Evaluate** — loss on held-out test set, inverse-transform predictions back to original MPG scale

## Results

The model converges cleanly — training and validation loss track closely, indicating no overfitting. Predictions are inverse-scaled back to real MPG values for interpretation.

| Stage | Detail |
|-------|--------|
| Training set | 80% (~314 cars) |
| Test set | 20% (~78 cars) |
| Architecture | 3-layer DNN (32 → 64 → 1) |
| Optimiser | RMSProp |
| Loss | MSE |
| Epochs | 100 |

## Setup

### Google Colab

Click the badge above — no setup required.

### Local

```bash
pip install tensorflow pandas scikit-learn matplotlib
git clone https://github.com/wsamuelw/regression-using-tensorflow.git
cd regression-using-tensorflow
jupyter notebook regression_using_tensorflow.ipynb
```

## Data

**Auto MPG Dataset** — from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Auto+MPG). ~400 cars with 7 features and 1 target.

| Feature | Type | Description |
|---------|------|------------|
| Cylinders | int | Number of cylinders |
| Displacement | float | Engine displacement |
| Horsepower | float | Engine power (6 missing, mean-imputed) |
| Weight | float | Vehicle weight |
| Acceleration | float | 0-60 mph time |
| Model Year | int | Year of manufacture |
| Origin | cat | 1=USA, 2=Europe, 3=Japan (one-hot encoded) |

**Target:** MPG (miles per gallon)

## Why Normalise?

Neural networks train faster and more reliably when inputs are on similar scales. Without normalisation, features like Weight (1,000–5,000) would dominate features like Acceleration (8–25), and the network would struggle to learn.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # mean=0, std=1
X_test = scaler.transform(X_test)        # use training stats
```

## Architecture

```
Input (9 features)
  ↓
Dense(32, activation='relu')
  ↓
Dense(64, activation='relu')
  ↓
Dense(1)  ← regression output (no activation)
```

No activation on the output layer — regression requires continuous output, not bounded to [0,1].

## Key Decisions

- **Mean imputation** for missing Horsepower — only 6 rows affected, mean is safe here
- **One-hot encoding** for Origin — categorical variable, ordinal encoding would imply USA < Europe < Japan
- **StandardScaler on labels** — helps the network converge; predictions are inverse-transformed back to MPG
- **RMSProp optimiser** — adaptive learning rate, works well out of the box for regression

## Tech Stack

- **TensorFlow / Keras** — neural network framework
- **scikit-learn** — StandardScaler, train/test split
- **pandas** — data loading and manipulation
- **matplotlib** — loss curve visualisation

## References

- [Auto MPG Dataset](http://archive.ics.uci.edu/ml/datasets/Auto+MPG)
- [TensorFlow regression tutorial](https://www.tensorflow.org/tutorials/keras/regression)
- [Keras Sequential API](https://www.tensorflow.org/guide/keras/sequential_model)

## License

MIT
