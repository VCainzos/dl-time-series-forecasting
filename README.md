# dl-time-series-forecasting
Deep Learning for Time Series Forecasting

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

Import Numpy, TensorFlow, and all preprocessing module dependencies:

```python
import numpy as np
import tensorflow as tf
from preprocessing.preprocessing import *
```
