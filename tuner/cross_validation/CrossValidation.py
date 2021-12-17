from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np

class CrossValidation():
    def __init__(self, epochs=5, batch_size=32, folds=3, shuffle=True):
        self.epochs=epochs
        self.batch_size=batch_size
        self.folds=folds
        self.shuffle=shuffle
        
    def build(self):
        self.kf=KFold(self.folds, shuffle=self.shuffle)
        
    def get_means(self, historial):
        means={}
        for key in historial[0].history.keys(): #Loop for metrics
            key_values=[]
            for i in historial: #Loop for histories
                key_values.append(i.history[key])
            means[key]=np.mean(key_values, axis=0)
        return means
        
    def __call__(self, model, dataset, *args, **kwargs):
    
        self.build() #Call build method
        
        samples = len([i for i in dataset.unbatch().batch(1)])

        x=next(iter(dataset.unbatch().batch(samples)))[0].numpy()
        y=next(iter(dataset.unbatch().batch(samples)))[1].numpy()

        historial=[]
        
        model_aux=model

        for fold, (train_indices, val_indices) in enumerate(self.kf.split(next(iter(dataset.unbatch().batch(samples)))[0])):
            
            model=model_aux
            print(f'Fold {fold}')
            x_train=tf.stack(x[train_indices,:,:])
            y_train=tf.stack(y[train_indices,:,:])
            dataset_train = tf.data.Dataset.from_tensors((x_train,y_train)).unbatch().batch(self.batch_size)

            x_val=tf.stack(x[val_indices,:,:])
            y_val=tf.stack(y[val_indices,:,:])
            dataset_val = tf.data.Dataset.from_tensors((x_val,y_val)).unbatch().batch(self.batch_size)

            history = model.fit(dataset_train, epochs=self.epochs, validation_data=dataset_val, *args, **kwargs)
            #historial.append(history.history['val_mean_absolute_error'])
            historial.append(history)
        return self.get_means(historial)