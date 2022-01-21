import copy
import plotly.graph_objects as go
import numpy as np


def boxplot(df, title="Data"):
    """Plot boxplots for a specified dataset

    :param df: dataset specified
    :type df: pandas.DataFrame
    :param title: title of the graph, defaults to "Data"
    :type title: str, optional
    :return: figure object
    :rtype: plotly.graph_object.Figure
    """

    fig = go.Figure()

    for col in df.columns:
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                line=dict(width=10),
                marker=dict(size=10),
                showlegend=False,
            )
        )

    if title == "R2":
        title = "R<sup>2</sup>"

    annotations = [
        dict(
            x=1,
            y=1,
            text=title,
            xref="paper",
            showarrow=False,
            yref="paper",
            xanchor="right",
            yanchor="top",
            bgcolor="white",
        )
    ]

    fig.update_layout(annotations=annotations)

    fig.update_layout(
        {
            "width": 3508,
            "height": 2480,
            "font": dict(size=80),
            "showlegend": False,
            "margin": dict(l=0, r=0, t=0, b=0),
        }
    )

    # fig.update_xaxes({"ticklabelposition": "inside"})
    # fig.update_yaxes({"ticklabelposition": "inside"})
    # fig.write_html("./images/normalization.html")
    # fig.show()
    return fig


def traces(df, split=0.8):
    """Plot traces for a specified dataset

    :param df: dataset specified
    :type df: pandas.DataFrame
    :param split: threshold for the training/test set, defaults to 0.9
    :type split: float, optional
    :return: figure object
    :rtype: plotly.graph_object.Figure
    """

    fig = go.Figure()

    n = len(df)
    for col in df.columns:

        # Training
        fig.add_traces(
            go.Scatter(
                x=df.index[0 : int(n * split)],
                y=df[col].values[0 : int(n * split)],
                name="",
                line=dict(width=10),
                # marker=dict(size=10),
                visible=True,
                showlegend=True,
            )
        )
        # Test
        fig.add_traces(
            go.Scatter(
                x=df.index[int(n * split) :],
                y=df[col].values[int(n * split) :],
                name=col,  # + " Test",
                line=dict(width=10),
                # marker=dict(size=10),
                visible=True,
                showlegend=True,
            )
        )

    # fig.update_layout({"title": "Data"})

    shapes = [
        dict(
            type="line",
            x0=df.index[int(n * split)],
            y0=0,
            x1=df.index[int(n * split)],
            y1=0.95,
            yref="paper",
            line=dict(color="Red", width=10, dash="dashdot"),
        ),
        dict(
            fillcolor="rgba(63, 81, 181, 0.2)",
            line={"width": 0},
            type="rect",
            x0=df.index[0],
            x1=df.index[int(n * split)],
            xref="x",
            y0=0,
            y1=0.95,
            yref="paper",
        ),
        dict(
            fillcolor="rgba(76, 175, 80, 0.1)",
            line={"width": 0},
            type="rect",
            x0=df.index[int(n * split)],
            x1=df.index[n - 1],
            xref="x",
            y0=0,
            y1=0.95,
            yref="paper",
        ),
    ]

    annotations = [
        dict(
            x=df.index[int(n * split / 2)],
            y=1,
            text="Training (" + str(int(split * 100)) + " %)",
            xref="x",
            showarrow=False,
            yref="paper",
        ),
        dict(
            x=df.index[int(n * (split + 1) / 2)],
            y=1,
            text="Test (" + str(round((1 - split) * 100)) + " %)",
            xref="x",
            showarrow=False,
            yref="paper",
        ),
    ]

    # Construct menus
    ncolumns = len(df.columns)

    # Create a list with traces visibilty
    visibility = [True] * ncolumns * 2
    train_visibility = [True, False] * ncolumns
    test_visibility = [False, True] * ncolumns

    updatemenus = [
        {
            #'active':1,
            "buttons": [
                {
                    "method": "update",
                    "label": "Training / Test",
                    "args": [
                        {"visible": test_visibility},
                        # 2. updates to the layout
                        {
                            "title": "Test Set",
                            "annotations": [[], annotations[1]],
                            "shapes": [[], [], shapes[2]],
                        },
                        # 3. which traces are affected
                        # All (default)
                    ],
                    "args2": [
                        {"visible": train_visibility},
                        # 2. updates to the layout
                        {
                            "title": "Training Set",
                            "annotations": [annotations[0], []],
                            "shapes": [[], shapes[1], []],
                        },
                        # 3. which traces are affected
                        # All (default)
                    ],
                },
                {
                    "method": "update",
                    "label": "Data",
                    "args": [
                        {"visible": visibility},
                        # 2. updates to the layout
                        {
                            "title": "Data",
                            "annotations": annotations,
                            "shapes": shapes,
                        },
                        # 3. which traces are affected
                        # All (default)
                    ],
                },
            ],
            "type": "buttons",
            #'type':'dropdown',
            "direction": "down",
            "showactive": True,
        }
    ]

    # update layout with buttons, and show the figure
    fig.update_layout(annotations=annotations, shapes=shapes)
    # fig.update_xaxes({"title": "Samples"})
    # fig.update_yaxes({"title": "Units"})
    fig.update_layout(
        {
            "width": 3508,
            "height": 2480,
            "font": dict(size=80),
            "margin": dict(l=0, r=0, t=0, b=0),
            "legend": dict(orientation="h", x=1, xanchor="right"),
        }
    )
    # fig.write_html("./images/data.html")
    # fig.show()
    return fig


def plot_predictions(window, group="test"):
    """Plot output predictions accroding to a specified window

    :param window: window object
    :type window: WindowGenerator
    :return: figure object
    :rtype: plotly.graph_object.Figure
    """

    if group == "test":
        set = window.test
        set_df = window.test_df
    elif group == "full":
        set = window.full
        set_df = window.full_df

    samples = [i for i in set.unbatch().batch(1)]  # make batches of one sample
    # It takes only the last batch (time_sequences, features). All samples of seq. within one batch
    *_, (inputs, labels) = iter(set.unbatch().batch(len(samples)))
    # Total windows=(samples-overlapping)/(window size-overlapping) -> stride=label_width
    samples = len(inputs) * (
        window.total_window_size
        - (window.total_window_size - window.label_width)
    )

    # Initialize figure
    fig = go.Figure()

    # Add Labels
    # (label_width=stride)
    labels = np.array(
        [
            set_df.index[n : n + window.label_width]
            for n in range(window.input_width, samples, window.label_width)
        ]
    ).flatten()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=set_df[window.label_columns[0]][labels],
            name='<SPAN STYLE="text-decoration:overline">'
            + window.label_columns[0]
            + "</SPAN>",
            mode="lines",
            line=dict(width=10, dash="solid"),
        )
    )

    # Add Predictions
    if window.models is not None:
        nmodels = len(window.models)
        for name, model in window.models.items():
            predictions = model(inputs).numpy().flatten()
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=predictions,
                    name=name + "<sub>" + str(window.input_width) + "</sub>",
                    # visible=False,
                    mode="lines",
                    line=dict(width=10),
                )
            )

        # fig.update_traces(visible=True, selector=dict(name=model.name))

        """fig.update_layout(
            title=dict(
                text=(
                    "Window horizon: "
                    + str(window.input_width)
                    + "I/"
                    + str(window.label_width)
                    + "O"
                )
            )
        )"""

    annotations = [
        dict(
            x=1,
            y=0.951,
            text="<i>Input/Label width: " + str(window.input_width) + "</i>",
            xref="paper",
            showarrow=False,
            yref="paper",
            xanchor="right",
            yanchor="top",
            bgcolor="white",
        )
    ]

    # Create a list with traces visibilty
    visibility = [True] + [False] * nmodels
    visible = visibility.copy()

    # Create a list with buttons
    buttons = []
    for index, name in enumerate(window.models):
        visible[index + 1] = True
        buttons.append(
            {
                "method": "update",
                "label": name,
                "args": [
                    {"visible": visible},
                ],
            }
        )
        visible = visibility.copy()

    # Create layout update
    updatemenus = [
        {
            #'active':1,
            "buttons": buttons,
            "type": "buttons",
            #           'type':'dropdown',
            "direction": "down",
            "showactive": True,
        }
    ]

    # fig.update_layout(annotations=annotations)
    fig.update_layout(
        {
            "width": 3508,
            "height": 2480,
            "font": dict(size=80),
            "margin": dict(l=0, r=0, t=0, b=0),
            "legend": dict(
                orientation="h", x=1, y=1, xanchor="right", yanchor="top"
            ),
        }
    )
    # fig.update_xaxes({"title": "Samples"})
    # fig.update_yaxes({"title": window.label_columns[0] + " (std)"})

    # fig.show()
    return fig


def plot_learning(tuner, title="Model"):
    """Plot learning curves during cross-validation registered in the hyperparameter
    optimization search by the tuner specified

    :param tuner: tuner object
    :type tuner: CvTuner
    :param title: title of the graph, defaults to "Model"
    :type title: str, optional
    :return: figure object
    :rtype: plotly.graph_object.Figure
    """

    fig = go.Figure()

    # Create a list with traces visibilty
    ntrials = len(tuner.cv_savings)
    visibility = [False] * ntrials * 3  # (Training+Validation+TableHp)

    # Flag to plot initially only the first trial traces
    flag = True

    # Create list to add traces of training and validation from each str(ntrial+1)
    traces = []

    # Create a list with buttons
    buttons = []

    for ntrial in range(ntrials):
        # Copy the list, otherwise both objetcs modify each other
        visible = visibility.copy()
        visible[ntrial * 3] = True
        visible[ntrial * 3 + 1] = True
        visible[ntrial * 3 + 2] = True

        if ntrial > 0:
            flag = False

        # Traces
        traces.append(
            go.Scatter(
                y=tuner.cv_savings[str(ntrial + 1)]["history"][
                    tuner.oracle.objective.name
                ],
                name="Training",
                mode="lines",
                visible=flag,
                line=dict(color="cyan", dash="solid"),
            )
        )
        traces.append(
            go.Scatter(
                y=tuner.cv_savings[str(ntrial + 1)]["history"][
                    "val_" + tuner.oracle.objective.name
                ],
                name="Validation",
                mode="lines",
                visible=flag,
                line=dict(color="green", dash="solid"),
            )
        )
        traces.append(
            go.Table(
                domain=dict(x=[0.2, 0.8]),
                visible=flag,
                header=dict(
                    values=[
                        i
                        for i in tuner.cv_savings[str(ntrial + 1)]["hp"].keys()
                    ],
                    line_width=0,
                ),
                cells=dict(
                    values=[
                        i
                        for i in tuner.cv_savings[str(ntrial + 1)][
                            "hp"
                        ].values()
                    ],
                    line_width=0,
                ),
            )
        )
        # Buttons
        buttons.append(
            {
                "method": "update",
                "label": "Trial " + str(ntrial + 1),
                "args": [{"visible": visible}],
            }
        )

    fig.add_traces(traces)

    updatemenus = [
        {
            #'active':1,
            "buttons": buttons,
            "type": "buttons",
            #'type':'dropdown',
            "direction": "down",
            "showactive": True,
        }
    ]

    fig.add_annotation(
        text="Hyperparameters set",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.1,
        showarrow=False,
    )

    # Metric name
    words = tuner.oracle.objective.name.split("_")
    if len(words) > 1:
        capital = []
        for word in words:
            capital.append(word[0].upper())
        name = "".join(capital)
    else:
        name = words[0][0].upper() + words[0][1:]

    fig.update_yaxes(title=name)
    fig.update_xaxes(title="Epochs")
    fig.update_layout({"title": title})
    fig.update_layout(updatemenus=updatemenus)

    fig.show()
    return fig


def plot_metrics(windows, metric_name, yrange=None, ytitle=""):
    """Plot bar chart with model metric results for a specified set of windows

    :param windows: set of windows specified
    :type windows: list
    :return: figure object
    :rtype: plotly.graph_object.Figure
    """

    multi_train_performance = {}
    multi_performance = {}

    for window in windows:
        multi_train_performance = dict(
            **multi_train_performance, **window.multi_train_performance
        )
        multi_performance = dict(
            **multi_performance, **window.multi_performance
        )

        for n, model_metrics in enumerate(
            window.multi_train_performance.values()
        ):
            if n == 0:
                m1 = list(model_metrics.keys())
                metrics = m1
            else:
                m2 = list(model_metrics.keys())
                metrics = [
                    m for m in m1 if m in set(m2)
                ]  # Find common metrics in models
                m1 = m2

    nmetrics = len(metrics)
    x = [key for key in multi_performance.keys()]  # model tags

    fig = go.Figure()
    visible = True

    # for metric_index, metric_name in enumerate(metrics):

    val = [
        v[metric_name] for v in multi_train_performance.values()
    ]  # Metrics could be in differents order
    test = [v[metric_name] for v in multi_performance.values()]

    # if metric_index != 0:
    #    visible = False

    # Names for the legend
    words = metric_name.split("_")
    capital = []
    for word in words:
        if word != "denor":
            capital.append(word[0].upper())
    name = "".join(capital)

    if name == "RS":
        name = "R<sup>2</sup>"

    fig.add_trace(
        go.Bar(
            name=name + "<sub>Train</sub>" + ytitle,
            y=val,
            x=x,
            marker_line_width=0,
        )
    )
    fig.add_trace(
        go.Bar(
            name=name + "<sub>Test</sub>" + ytitle,
            y=test,
            x=x,
            marker_line_width=0,
        )
    )

    if range != None:
        fig.update_yaxes(range=yrange)
    """else:
        fig.update_yaxes(dict(ticklabelposition="inside"))"""

    # Create a list with traces visibilty
    visibility = [False] * nmetrics * 2
    # Copy the list, otherwise both objetcs modify each other
    visible = visibility.copy()

    # Create a list with buttons
    buttons = []
    for nmetric in range(nmetrics):
        visible[nmetric * 2] = True
        visible[nmetric * 2 + 1] = True

        # Metric name
        words = metrics[nmetric].split("_")
        if len(words) > 1:
            capital = []
            for word in words:
                if word != "denor":
                    capital.append(word[0].upper())
                else:
                    capital.append(" (" + window.label_columns[0] + ")")

            name = "".join(capital)
        else:
            name = words[0][0].upper() + words[0][1:]

        """if nmetric == 0:
            fig.update_yaxes(dict(title=name))"""

        buttons.append(
            {
                "method": "update",
                "label": name,
                "args": [
                    {"visible": visible},
                    {"yaxis": {"title": name, "ticklabelposition": "inside"}},
                ],
            }
        )
        visible = visibility.copy()

    updatemenus = [
        {
            #'active':1,
            "buttons": buttons,
            "type": "buttons",
            #'type':'dropdown',
            "direction": "down",
            "showactive": True,
        }
    ]

    fig.update_layout(
        {
            "width": 3508,
            "height": 2480,
            "font": dict(size=80),
            "margin": dict(l=0, r=0, t=0, b=0),
            "legend": dict(
                orientation="h",
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
            ),
        }
    )

    # fig.update_xaxes(title="Models")
    # fig.update_layout({"title": "Metrics"})
    # fig.update_layout(updatemenus=updatemenus)

    # fig.show()
    return fig
