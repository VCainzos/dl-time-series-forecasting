import keras_tuner as kt
from .cross_validation.CrossValidation import CrossValidation

class MyTuner(kt.engine.tuner.Tuner):
    def __init__(self, window, cv_args=dict(), **kwargs):
        self.window=window
        self.cv_args=cv_args             
        kwargs['hypermodel']=kwargs['hypermodel'](self.window) #Pass window to hypermodel function callable
        super().__init__(**kwargs)
        
    def save_cv(self, history, hp):
        self.cv_savings = getattr(self, 'cv_savings', {})
        self.cv_savings[str(self._display.trial_number)]=dict(history=history, hp=hp) #Add metrics for the trial number

    #Not necessary as long as MyTuner() object is called throughout bounded methods
    #def __call__(self):
    #    self.cv_savings={} #Create a dict to save cv mean histories for trial
    
    def _build_and_fit_model(self, trial, *args, **kwargs): #*args and **kwargs are passed by 'search' to get the callbacks within for saving the model results
        #batch_size=32
        """
        Args:
            trial: A `Trial` instance that contains the information needed to
                run this trial. `Hyperparameters` can be accessed via
                `trial.hyperparameters`.
            *args: Positional arguments passed by `search`.
            **kwargs: Keyword arguments passed by `search`.
        Returns:
            The fit history.
        """
        hp = trial.hyperparameters
        model = self._try_build(hp)
        batch_size=hp.Int('batch_size', 32, 40, step=2) #Manual update to the search space including batch-size variable
        #Implementation of cross-validation
        cv=CrossValidation(batch_size=batch_size, **self.cv_args)
        history_means=cv(model, self.window.train, *args, **kwargs) #Here are passed the callbacks within *args and **kwargs
        self.save_cv(history_means, hp.values)
        return history_means