import keras_tuner as kt
from .cross_validation.CrossValidation import CrossValidation


class MyTuner(kt.engine.tuner.Tuner):
    """Subclass of Tuner from KerasTuner to implement cross-validation

    Args:
        kt.engine.tuner.Tuner (kt.engine.tuner.Tuner): inheritance of the kt.engine.tuner.Tuner
    """

    def __init__(self, window, cv_args=dict(), **kwargs):
        """Define new customized attributes

        Args:
            window (WindowGenerator): window object
            cv_args (dict, optional): dictionary with additional parameters to configure cross-validation. Defaults to dict().
        """
        self.window = window
        self.cv_args = cv_args
        # Pass window to hypermodel function callable
        kwargs["hypermodel"] = kwargs["hypermodel"](self.window)
        super().__init__(**kwargs)

    def save_cv(self, history, hp):
        """Saves cross-validation results and hyperparameters for the current trial

        Args:
            history (dict): dictionary with metrics and mean results
            hp (kt.engine.hyperparameters.HyperParameters): set of hyperparameters used for the current trial
        """
        self.cv_savings = getattr(self, "cv_savings", {})
        # Add metrics for the trial number
        self.cv_savings[str(self._display.trial_number)] = dict(history=history, hp=hp)

    # Not necessary as long as MyTuner() object is called throughout bounded methods
    # def __call__(self):
    #    self.cv_savings={} #Create a dict to save cv mean histories for trial

    def _build_and_fit_model(self, trial, *args, **kwargs):
        """Implements the creation and fitting of the model

        Args:
            trial (kt.engine.trial.Trial): A `Trial` instance that contains the information needed to run this trial. `Hyperparameters` can be accessed via `trial.hyperparameters`

        Returns:
            history_means (dict): dictionary with metrics and results
        """

        # *args and **kwargs are passed by 'search' to get the callbacks within for saving the model results
        # batch_size=32

        hp = trial.hyperparameters
        model = self._try_build(hp)
        # Manual update to the search space including batch-size variable
        batch_size = hp.Int("batch_size", 32, 40, step=2)
        # Implementation of cross-validation
        cv = CrossValidation(batch_size=batch_size, **self.cv_args)
        # Here are passed the callbacks within *args and **kwargs
        history_means = cv(model, self.window.train, *args, **kwargs)
        self.save_cv(history_means, hp.values)
        return history_means
