# dl-time-series-forecasting

This repository is an easy-to-use framework which contains util dependencies that may
help out creating deep learning models as well as optimizing and comparing each other results
for a time series analysis. Easily configure your sliding window with a define-by-run syntax
and build the dataset according to a specified size and prediction horizon.

---
## Installation

It is necessary for some specific packages to be installed from which the framework inherits functionlaities:

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

- The visualization module
- WindowGenerator class
- Wrapper function that creates and returns a Keras model.
- MyTuner class
- KerasTuner package

```python
import visualization.visualization as vs
from window.WindowGenerator import *
from models.lstm import *
from tuner.Tuner import MyTuner
import keras_tuner as kt
```

Once all aforementioned dependencies have been imported, the following steps can be taken. First, create a window object.
`input_width` figures out the window size and `label_width` the prediction horizon.
```python
window = WindowGenerator(
    dataframe,
    input_width=5,
    label_width=10,
    shift=10,
    label_columns=['output_variable'])
``` 
Then, create a tuner object of the customized class MyTuner which will be dealing with the hyperparameter optimization.
`objective` specifies the objective to select the best models, and `max_trials` to the number of different models to try.
```python
tuner = MyTuner(
    oracle=kt.oracles.RandomSearch(
        objective=kt.Objective("mean_squared_error", "min"), max_trials=5
    ),
    hypermodel=build_model_lstm,
    window=window
)
```
Perform the search (i.e., a `RandomSearch`) and build a model with the best configuration.
```python
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model_lstm = tuner.hypermodel.build(best_hps)
```
The resulting model is ready to be fitted on the full training set and evaluated on the test set. Thereby, predictions can
be ploted using the `vs` module functionalities.
