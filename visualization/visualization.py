import plotly.graph_objects as go

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