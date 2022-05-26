from functools import cache
from dash import dcc, html, Input, Output, callback
import pandas as pd
import pathlib
import plotly.graph_objects as go
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
import numpy as np

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
data_cr_wide = pd.read_csv(DATA_PATH.joinpath("data_cr_wide.csv"), index_col='month', sep=";")
clusters_all = [x for x in range(0,6)]


@cache
def get_model(n_clusters, metric, method):
    if metric == 'softdtw':
        max_iter = 25
    else: 
        max_iter = 50

    x = data_cr_wide.transpose().values
    if method == 'TimeSeriesKMeans':
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=max_iter, n_init=2, random_state=11).fit(x)
    if method == 'KernelKMeans':
        model = KernelKMeans(n_clusters=n_clusters, kernel="gak", max_iter=25, n_init=2).fit(x)
    if method == 'KShape':
        model = KShape(n_clusters=n_clusters, max_iter=25, n_init=4).fit(x)
    return model


layout = html.Div([
    html.H3('Clustering Categories by Seasonality of Conversion Rate'),
    html.H6('Choose a clustering method:'),
    dcc.RadioItems(
    options=[
        {'label': 'TimeSeriesKMeans', 'value': 'TimeSeriesKMeans'},
        {'label': 'KernelKMeans', 'value': 'KernelKMeans'},
        {'label': 'KShape', 'value': 'KShape'},
    ],
    value='TimeSeriesKMeans', inline=False, id='radio_cr_method'
    ),
    html.H6('Set metric for TimeSeriesKMeans:'),
    dcc.RadioItems(
    options=[
        {'label': 'euclidean distance', 'value': 'euclidean'},
        {'label': 'dynamic time warping (dtw)', 'value': 'dtw'},
        {'label': 'soft dynamic time warping (soft dtw)', 'value': 'softdtw'},
    ],
    value='euclidean', inline=False, id='radio_cr_metric'
    ),
    html.H6('Set number of clusters:'),
    dcc.Slider(2, 12, 1, value=6, id='slider_cr'),
    html.H6('Choose a Cluster:'),
    dcc.Slider(min=0, max=5, step=1, value=0, id='slider_cr_cluster'),
    dcc.RadioItems(
    options=[
    {'label': " straight lines ", 'value': 'linear'},
    {'label': " spaghetti ", 'value': 'spline'},
    ],
    value='linear', inline=True, id='radio_lines'
    ),
    dcc.Graph(id="graph_cr"),
])


@callback(
    Output(component_id='graph_cr', component_property='figure'),
    Output(component_id='slider_cr_cluster', component_property='max'),
    Input(component_id='slider_cr_cluster', component_property='value'),
    Input(component_id='radio_cr_metric', component_property='value'),
    Input(component_id='radio_cr_method', component_property='value'),
    Input(component_id='slider_cr', component_property='value'),
    Input(component_id='radio_lines', component_property='value'),
)
def update_chart(cluster_number, metric, method, n_clusters, line_shape):
    data = data_cr_wide
    df_cluster = pd.DataFrame(list(zip(data.columns, get_model(n_clusters, metric, method).labels_)), columns=['category', 'cluster'])
    # make dictionaries for quality measuring
    cluster_cat_dict = df_cluster.groupby(['cluster'])['category'].apply(lambda x: [x for x in x]).to_dict()
    cluster_len_dict = df_cluster['cluster'].value_counts().to_dict()

    # Prevents error when decreasing n_clusters and previous cluster_number of graph is higher
    try: cluster_cat_dict[cluster_number]
    except: cluster_number = 0

    # get quality score based on the correlation between categories in the cluster
    x_corr = data_cr_wide[cluster_cat_dict[cluster_number]].corr().abs().values
    x_corr_mean = round(x_corr[np.triu_indices(x_corr.shape[0],1)].mean(),2)

    # plot all categories in one cluster
    if method == "TimeSeriesKMeans":
        plot_title = f'{method} cluster {cluster_number} (quality={x_corr_mean}, n={cluster_len_dict[cluster_number]}, metric={metric})'
    else:
        plot_title = f'{method} cluster {cluster_number} (quality={x_corr_mean}, n={cluster_len_dict[cluster_number]})'
    cols = cluster_cat_dict[cluster_number]
    fig = go.Figure()
    ind = data.index
    for i, col in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=ind, y=data[col], name=col, line={'width':1}, hoverlabel={'namelength':-1}, showlegend=True, line_shape=line_shape
            )
        )    
    fig.update_xaxes(title_text="months", dtick=[1,len(data.index)+1])
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(title_text=plot_title, template="plotly_dark", height=600)
    return fig, n_clusters-1
