import plotly.graph_objects as go
import numpy as np

def boxplot(df, title='Data Standarized'):

    fig = go.Figure()

    #Concat train_df +
    for col in df.columns:
        fig.add_trace(go.Box(y=df[col], name=col))

    fig.update_layout({'title' : title})
    fig.update_xaxes({'title':'Variables'})
    fig.update_yaxes({'title':'Values'})
    #fig.write_html("./images/normalization.html")
    fig.show()

def traces(df, split=0.9):
    fig = go.Figure()

    n = len(df)
    for col in df.columns:

        #Training
        fig.add_traces(go.Scatter(x=df.index[0:int(n*split)], y=df[col].values[0:int(n*split)], name=col+' Training', visible=True, showlegend=True))
        #Test
        fig.add_traces(go.Scatter(x=df.index[int(n*split):], y=df[col].values[int(n*split):], name=col+' Test', visible=True, showlegend=True))

    fig.update_layout({'title' : 'Data'})

    shapes=[
            dict(
                type="line",
                x0=int(n*split), y0=0, x1=int(n*split), y1=0.95, yref='paper',
                line=dict(color="Red",width=3,dash='dashdot')
            ),
            dict(
                fillcolor="rgba(63, 81, 181, 0.2)",
                line={"width": 0},
                type="rect",
                x0=0,
                x1=int(n*split),
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            ),
            dict(
                fillcolor="rgba(76, 175, 80, 0.1)",
                line={"width": 0},
                type="rect",
                x0=int(n*split),
                x1=n,
                xref="x",
                y0=0,
                y1=0.95,
                yref="paper"
            )]

    annotations=[
            dict(
                x=int(n*split/2),
                y=1.1,
                text="Training ("+str(int(split*100))+ " %)",
                xref="x",
                showarrow=False,
                yref="paper"
            ),
            dict(
                x=int(n*(split+1)/2),
                y=1.1,
                text="Test ("+str(round((1-split)*100))+ " %)",
                xref="x",
                showarrow=False,
                yref="paper"
            )]

    #Construct menus
    ncolumns = len(df.columns)

    #Create a list with traces visibilty
    visibility=[True]*ncolumns*2
    train_visibility = [True, False]*ncolumns
    test_visibility = [False, True]*ncolumns


    updatemenus = [{
    #               'active':1,
                    'buttons': [{'method': 'update',
                                'label': 'Training / Test',
                                'args': [
                                            
                                        {'visible': test_visibility},
                                        
                                        # 2. updates to the layout
                                        {'title': 'Test Set', 'annotations':[[], annotations[1]], 'shapes':[[], [], shapes[2]]},
                                        
                                        # 3. which traces are affected
                                        #All (default)
                                        ],
                                'args2': [
                                            
                                        {'visible': train_visibility},
                                        
                                        # 2. updates to the layout
                                        {'title': 'Training Set', 'annotations':[annotations[0], []], 'shapes':[[], shapes[1], []]},
                                        
                                        # 3. which traces are affected
                                        #All (default)
                                        ]
                                },
                                {'method': 'update',
                                'label': 'Data',
                                'args': [
                                            
                                        {'visible': visibility},
                                        
                                        # 2. updates to the layout
                                        {'title': 'Data', 'annotations':annotations, 'shapes':shapes},
                                        
                                        # 3. which traces are affected
                                        #All (default)
                                        ],
                                }
                                ],
                    'type':'buttons',
    #               'type':'dropdown',
                    'direction': 'down',
                    'showactive': True,}]


    # update layout with buttons, and show the figure
    fig.update_layout(updatemenus=updatemenus, annotations=annotations, shapes=shapes)
    fig.update_xaxes({'title':'Samples'})
    fig.update_yaxes({'title':'Units'})
    #fig.write_html("./images/data.html")
    fig.show()

def plot_predictions(window):
    samples=[i for i in window.test.unbatch().batch(1)] #make batches of one sample
    *_, (inputs, labels) = iter(window.test.unbatch().batch(len(samples)))#It takes only the last batch (time_sequences, features). All samples of seq. within one batch
    samples=len(inputs)*(window.total_window_size-(window.total_window_size-window.label_width)) #Total windows=(samples-overlapping)/(window size-overlapping)

    # Initialize figure
    fig = go.Figure()

    #Add Labels
    labels = np.array([window.test_df.index[n:n+window.label_width] for n in range(window.input_width, samples, window.label_width)]).flatten() #label_width=stride
    fig.add_trace(
        go.Scatter(x=labels, 
                y=window.test_df[window.label_columns[0]][labels],
                name="Real",
                mode='lines',
                line=dict(color="cyan", dash='solid')))
    
    #Add Predictions
    if window.models is not None:
        nmodels=len(window.models)
        for name, model in window.models.items():
            predictions = model(inputs).numpy().flatten()
            fig.add_trace(
            go.Scatter(x=labels, 
                    y=predictions,
                    name=name,
                    visible=False,
                    mode='lines',
                    line=dict(color="orange")))
        
        #fig.update_traces(visible=True, selector=dict(name=model.name))

        fig.update_layout( title_text='Window horizon: '+str(window.input_width)+'I/'+str(window.label_width)+'O')

    #Create a list with traces visibilty
    visibility=[True]+[False]*nmodels
    visible=visibility.copy()

    #Create a list with buttons
    buttons=[]
    for index, name in enumerate(window.models):
        visible[index+1]=True
        buttons.append({'method': 'update',
                            'label': name,
                            'args': [
                                    {'visible': visible},
                                    ]
                        })
        visible=visibility.copy()

    #Create layout update
    updatemenus=[{
#                 'active':1,
            'buttons': buttons,
            'type':'buttons',
#           'type':'dropdown',
            'direction': 'down',
            'showactive': True,}]

    fig.update_layout(updatemenus=updatemenus)
    fig.update_xaxes({'title':'Samples'})
    fig.update_yaxes({'title':window.label_columns[0]+' (std)'})

    fig.show()
    return 
    
def plot_learning(tuner, title='Model'):

        fig = go.Figure()

        #Create a list with traces visibilty
        ntrials=len(tuner.cv_savings)
        visibility=[False]*ntrials*3 #(Training+Validation+TableHp)
        
        #Flag to plot initially only the first trial traces
        flag=True 

        #Create list to add traces of training and validation from each str(ntrial+1)
        traces=[]

        #Create a list with buttons
        buttons=[]

        for ntrial in range(ntrials):
                visible=visibility.copy() #Copy the list, otherwise both objetcs modify each other
                visible[ntrial*3]=True
                visible[ntrial*3+1]=True
                visible[ntrial*3+2]=True
                
                if ntrial > 0:
                        flag=False

                #Traces
                traces.append(
                        go.Scatter(y=tuner.cv_savings[str(ntrial+1)]['history'][tuner.oracle.objective.name],
                        name="Training",
                        mode='lines',
                        visible=flag,
                        line=dict(color="cyan", dash='solid'))
                )
                traces.append(
                        go.Scatter(y=tuner.cv_savings[str(ntrial+1)]['history']['val_'+tuner.oracle.objective.name],
                        name="Validation",
                        mode='lines',
                        visible=flag,
                        line=dict(color="green", dash='solid'))
                )
                traces.append(
                        go.Table(domain=dict(x=[0.2,0.8]), 
                        visible=flag,
                        header=dict(values=[i for i in tuner.cv_savings[str(ntrial+1)]['hp'].keys()], line_width=0),
                        cells=dict(values=[i for i in tuner.cv_savings[str(ntrial+1)]['hp'].values()], line_width=0))
                )
                #Buttons
                buttons.append({
                        'method': 'update',
                        'label': 'Trial '+str(ntrial+1),
                        'args': [                                     
                                {'visible': visible}
                                ]
                        })
                
        fig.add_traces(traces)
        
        updatemenus =[{
        #       'active':1,
                'buttons': buttons,
                'type':'buttons',
        #       'type':'dropdown',
                'direction': 'down',
                'showactive': True
                }]

        fig.add_annotation(
                text="Hyperparameters set",
                xref="paper", yref="paper",
                x=0.5, y=1.1, showarrow=False
                )
        
        #Metric name
        words=tuner.oracle.objective.name.split('_')
        if len(words)>1:
            capital=[]
            for word in words:
                capital.append(word[0].upper())
            name=''.join(capital)
        else:
            name=words[0][0].upper()+words[0][1:]

        fig.update_yaxes(title=name)
        fig.update_xaxes(title='Epochs')
        fig.update_layout({'title': title})
        fig.update_layout(updatemenus=updatemenus)

        fig.show()
        return fig

def plot_metrics(windows, metrics = ['loss', 'mean_absolute_error', 'mean_squared_error', 'mean_absolute_error_denor']):
    
    multi_train_performance = {}
    multi_performance = {}

    for window in windows:
        multi_train_performance = dict(**multi_train_performance, **window.multi_train_performance)
        multi_performance = dict(**multi_performance, **window.multi_performance)

    nmetrics = len(metrics)
    x = [key for key in multi_performance.keys()]

    fig = go.Figure()
    visible = True
    
    for metric_index, metric_name in enumerate(metrics):
    
        val_mae = [v[metric_index] for v in multi_train_performance.values()]
        test_mae = [v[metric_index] for v in multi_performance.values()]

        if metric_index!=0:
            visible=False
    
        fig.add_trace(go.Bar(name='Training', y=val_mae, x=x, visible=visible))
        fig.add_trace(go.Bar(name='Test', y=test_mae, x=x, visible=visible))

    fig.update_yaxes(dict(ticklabelposition='inside'))

    #Create a list with traces visibilty
    visibility=[False]*nmetrics*2
    visible=visibility.copy() #Copy the list, otherwise both objetcs modify each other

    #Create a list with buttons
    buttons=[]
    for nmetric in range(nmetrics):
        visible[nmetric*2]=True
        visible[nmetric*2+1]=True
              
        #Metric name
        words=metrics[nmetric].split('_')
        if len(words)>1:
            capital=[]
            for word in words:
                capital.append(word[0].upper())
            name=''.join(capital)
        else:
            name=words[0][0].upper()+words[0][1:]

        buttons.append({'method': 'update',
                        'label': name,
                        'args': [
                                 {'visible': visible},
                                 {'yaxis': {'ticklabelposition': 'inside'}}
                                ],
                        })
        visible=visibility.copy()
    
    updatemenus = [{
#               'active':1,
                'buttons': buttons,
                'type':'buttons',
#               'type':'dropdown',
                'direction': 'down',
                'showactive': True}]

    fig.update_xaxes(title= 'Models')
    fig.update_layout({'title':'Metrics'})
    fig.update_layout(updatemenus=updatemenus)

    fig.show()
    return fig
