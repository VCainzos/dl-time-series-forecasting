import keras_tuner as kt
from .cross_validation.CrossValidation import CrossValidation


class CvTuner(kt.engine.tuner.Tuner):
    """Subclass of Tuner from KerasTuner to implement cross-validation"""

    def __init__(self, window, cv_args=dict(), **kwargs):
        """Defines new customized attributes

        :param window: window object
        :type window: WindowGenerator
        :param cv_args: additional parameters to configure cross-validation, defaults to dict()
        :type cv_args: dict, optional
        """

        self.window = window
        self.cv_args = cv_args
        # Pass window to hypermodel function callable
        kwargs["hypermodel"] = kwargs["hypermodel"](self.window)
        super().__init__(**kwargs)

    def save_cv(self, history, kscores, hp, trial):
        """Saves cross-validation and hyperparameters for the current trial

        :param history: dictionary with metrics and mean results
        :type history: dict
        :param hp: set of hyperparameters used for the current trial
        :type hp: kt.engine.hyperparameters.HyperParameters
        """

        self.cv_savings = getattr(self, "cv_savings", {})
        self.fold_savings = getattr(self, "fold_savings", {})

        # Add metrics for the trial number
        self.cv_savings[str(self._display.trial_number)] = dict(
            history=history, hp=hp
        )
        self.fold_savings[trial.trial_id] = dict(folds=kscores)

    # Not necessary as long as MyTuner() object is called throughout bounded methods
    # def __call__(self):
    #    self.cv_savings={} #Create a dict to save cv mean histories for trial

    def _build_and_fit_model(self, trial, *args, **kwargs):
        """Implemets the creation and fitting of the model

        :param trial: A `Trial` instance that contains the information needed to run this trial.
        `Hyperparameters` can be accessed via `trial.hyperparameters`
        :type trial: kt.engine.trial.Trial
        :return: dictionary with metrics and results
        :rtype: dict
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
        history_means, kscores = cv(model, self.window.train, *args, **kwargs)
        self.save_cv(history_means, kscores, hp.values, trial)
        return history_means
