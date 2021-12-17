# dl-time-series-forecasting

This repository is an easy-to-use framework which contains util dependencies that may
help out creating deep learning models as well as optimizing and comparing each other results
for a time series analysis. Easily configure your sliding window with a define-by-run syntax
and build the dataset according to a specified size and prediction horizon.

---
## Installation

It is necessary for some packages to be installed from which the framework inherits functionlaities:

- KerasTuner

[KerasTuner](https://keras.io/keras_tuner/) is an easy-to-use, scalable hyperparameter optimization library that solves the pain points of hyperparameter search.

KerasTuner requires **Python 3.6+** and **TensorFlow 2.0+**.

Install the latest release:

```
pip install keras-tuner --upgrade
```

You can also check out other versions [here](https://github.com/keras-team/keras-tuner).

- Plotly

[plotly.py](https://plot.ly/python) is an interactive, open-source, and browser-based graphing library for Python

```
pip install plotly==5.4.0`
```

---
## Quick introduction

Import the following libraries:

- visualization module:
```python
import visualization.visualization as vs
```
- WindowGenerator class
```python
from window.WindowGenerator import *
```
- A function already defined that creates and returns a Keras model.
Using the `hp` argument to define the hyperparameters during model creation.
```python
from models.lstm import *
```
-MyTuner class
```python
from tuner.Tuner import MyTuner
```
-KerasTuner package
```python
import keras_tuner as kt
```
Create a window object
```python
window = WindowGenerator(
    DataFrame,
    input_width=10,
    label_width=10,
    shift=10,
    label_columns=['output_variable'])

Craete a tuner object of the customized class MyTuner
```python
tuner = MyTuner(
    oracle=kt.oracles.RandomSearch(
        objective=kt.Objective("mean_squared_error", "min"), max_trials=1
    ),
    hypermodel=build_model_lstm,
    window=window
)
