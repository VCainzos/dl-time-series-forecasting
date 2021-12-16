import numpy as np
import tensorflow as tf
from preprocessing.preprocessing import *

class WindowGenerator():
    def __init__(self, df, input_width, label_width, shift, label_columns=None, **kwargs):
        
        self.set_batch_size() #Initialize the batch_size
        # Store the raw data.
        train, test = split(df, **kwargs)
        # And standarized dataframes
        self.train_df, self.test_df = standarize(train, test)
        #These both wil be used in custom metrics as variables
        self.train_std = train.std() 
        self.train_mean = train.std()
        #self.val_df = val_df

        # Work out the label column-indices as pairs key-value of a dictionary.
        self.label_columns = label_columns #(used in split_window)
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}

        # Get the features-indices as pairs key-value of a dictionary (used in function split_window)
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width #Number of input values
        self.label_width = label_width #Number of predictions
        self.shift = shift #This is the difference between the last input and label indices (time into the future -> offset)

        self.total_window_size = input_width + shift #(not all the indices through the window must be used in every case)

        self.input_slice = slice(0, input_width) #Returns a slice object representing the set of indices specified by range(start,stop,step)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice] #Create an array of slicing input indices

        self.label_start = self.total_window_size - self.label_width #Get the label indices
        self.labels_slice = slice(self.label_start, None) #Create slice object
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice] #Create an array of slicing label indices

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        #Splits a raw tensor of samples in inputs and labels using arrays of indices
        inputs = features[:,self.input_slice, :] #Split tensor using inputs indices across samples dimension (Could it be )
        labels = features[:,self.labels_slice, :] #Split tensor using label indices across samples dimension
        if self.label_columns is not None:
            #Split tensor using label column name across features dimension
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot_predictions(self, models=None, plot_col='Capacity'):
        samples=[i for i in self.test.unbatch().batch(1)] #make batches of one sample
        #inputs, labels = next(iter(self.test.unbatch().batch(len(samples))))
        *_, (inputs, labels) = iter(self.test.unbatch().batch(len(samples)))#It takes only the last batch (time_sequences, features). All samples of seq. within one batch
        samples=len(inputs)*(self.total_window_size-(self.total_window_size-self.label_width)) #Total windows=(samples-overlapping)/(window size-overlapping)
        d = len(self.test_df)-samples #delay of samples which are not enough to slide the window

        # Initialize figure
        fig = go.Figure()

        #Add Labels
        labels = np.array([self.test_df.index[n:n+self.label_width] for n in range(self.input_width, samples, self.label_width)]).flatten() #label_width=stride
        fig.add_trace(
            go.Scatter(x=labels, 
                    y=self.test_df[plot_col][labels],
                    name="Real",
                    mode='lines',
                    line=dict(color="cyan", dash='solid')))
        
        #Add Predictions
        if models is not None:
            nmodels=len(models)
            for model in models:
                predictions = model(inputs).numpy().flatten()
                fig.add_trace(
                go.Scatter(x=labels, 
                        y=predictions,
                        name=model.name,
                        visible=False,
                        mode='lines',
                        line=dict(color="orange")))
            
            #fig.update_traces(visible=True, selector=dict(name=model.name))

            fig.update_layout( title_text='Window horizon: '+str(self.input_width)+'I/'+str(self.label_width)+'O')

        #Create a list with traces visibilty
        visibility=[True]+[False]*nmodels
        visible=visibility.copy()

        #Create a list with buttons
        buttons=[]
        for nmodel in range(1, nmodels):
            visible[nmodel]=True
            buttons.append({'method': 'update',
                             'label': models[nmodel].name,
                             'args': [
                                       {'visible': visible},
                                     ],
                              })
            visible=visibility.copy()

        #Create layout update
        updatemenus=[{
#                 'active':1,
                'buttons': buttons,
                'type':'buttons',
#               'type':'dropdown',
                'direction': 'down',
                'showactive': True,}]

        fig.update_layout(updatemenus=updatemenus)
        fig.update_xaxes({'title':'Samples'})
        fig.update_yaxes({'title':'Capacity (std)'})

        fig.show()
        return fig
        
    def set_batch_size(self, batch_size=32):
        self.batch_size=batch_size
        
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width, #Using label width as stride to avoid overlapping predictions
            shuffle=False, #Doing shuffle here would cause the loss of data sequence missing the tendency, this is not desired
            batch_size=self.batch_size)#len(data),)#Number of time sequences in each batch, in this case only one batch is used 

        ds = ds.map(self.split_window)

        return ds #Dataset = batch/'s of tensors
    
    @property
    def train(self):
        ds = self.make_dataset(self.train_df)
        return ds.shuffle(len(ds), reshuffle_each_iteration=False) #Shuffle data before building the Dataset

    #@property
    #def val(self):
        #return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)