"""
Dash-app page for clustering page-impressions data
"""
# pylint: disable=C0301
import pathlib
from functools import cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from tslearn.clustering import KernelKMeans, KShape, TimeSeriesKMeans

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
data_pi_wide = pd.read_csv(
    DATA_PATH.joinpath("data_pi_wide.csv"), index_col="month", sep=";"
)
clusters_all = [x for x in range(0, 6)]


def normalize_df(dataframe):
    """normalizes the whole dataframe"""
    df_ = dataframe.reset_index()
    df_norm = (df_ - df_.min()) / (df_.max() - df_.min())
    df_norm.drop(["month"], axis=1, inplace=True)
    df_norm = pd.concat((df_norm, df_.month), axis=1)
    df_norm.set_index("month", inplace=True)
    return df_norm


@cache
def get_model(n_clusters, metric, norm, method):
    """loads model-type and mormalizes the data if necessary"""
    if metric == "softdtw":
        max_iter = 25
    else:
        max_iter = 50
    if norm is True:
        data = normalize_df(data_pi_wide).transpose().values
    else:
        data = data_pi_wide.transpose().values
    if method == "TimeSeriesKMeans":
        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric=metric,
            max_iter=max_iter,
            n_init=2,
            random_state=11,
        ).fit(data)
    elif method == "KernelKMeans":
        model = KernelKMeans(
            n_clusters=n_clusters, kernel="gak", max_iter=25, n_init=2
        ).fit(data)
    else:  # method == "KShape"
        model = KShape(n_clusters=n_clusters, max_iter=25, n_init=4).fit(data)
    return model


layout = html.Div(
    [
        html.H3("Clustering Categories by Seasonality of Page Impressions"),
        html.H6("Choose a clustering method:"),
        dcc.RadioItems(
            options=[
                {"label": "TimeSeriesKMeans", "value": "TimeSeriesKMeans"},
                {"label": "KernelKMeans", "value": "KernelKMeans"},
                {"label": "KShape", "value": "KShape"},
            ],
            value="TimeSeriesKMeans",
            inline=False,
            id="radio_pi_method",
        ),
        html.H6("Set metric for TimeSeriesKMeans:"),
        dcc.RadioItems(
            options=[
                {"label": "euclidean distance", "value": "euclidean"},
                {"label": "dynamic time warping (dtw)", "value": "dtw"},
                {"label": "soft dynamic time warping (soft dtw)", "value": "softdtw"},
            ],
            value="euclidean",
            inline=False,
            id="radio_pi_metric",
        ),
        html.H6("Preprocess data:"),
        dcc.RadioItems(
            options=[
                {"label": "not normalized", "value": False},
                {"label": "normalized", "value": True},
            ],
            value=False,
            inline=False,
            id="radio_pi_norm",
        ),
        html.H6("Set number of clusters:"),
        dcc.Slider(2, 12, 1, value=6, id="slider_pi"),
        html.H6("Choose a Cluster:"),
        dcc.Slider(min=0, max=5, step=1, value=0, id="slider_pi_cluster"),
        dcc.RadioItems(
            options=[
                {"label": " straight lines ", "value": "linear"},
                {"label": " spaghetti ", "value": "spline"},
            ],
            value="linear",
            inline=True,
            id="radio_lines",
        ),
        dcc.Graph(id="graph_pi"),
    ]
)


@callback(
    Output(
        component_id="graph_pi", component_property="figure"
    ),  # output: fig into graph_pi
    Output(
        component_id="slider_pi_cluster", component_property="max"
    ),  # output: clusters_all into dropdown_pi "options"
    Input(component_id="slider_pi_cluster", component_property="value"),
    Input(component_id="radio_pi_metric", component_property="value"),
    Input(component_id="radio_pi_norm", component_property="value"),
    Input(component_id="radio_pi_method", component_property="value"),
    Input(component_id="slider_pi", component_property="value"),
    Input(component_id="radio_lines", component_property="value"),
)
def update_chart(cluster_number, metric, norm, method, n_clusters, line_shape):
    """Dash callback-function to update the graph"""
    if norm is True:
        data = normalize_df(data_pi_wide)
        nor = "normalized"
    else:
        data = data_pi_wide
        nor = "not normalized"
    df_cluster = pd.DataFrame(
        list(zip(data.columns, get_model(n_clusters, metric, norm, method).labels_)),
        columns=["category", "cluster"],
    )
    # make dictionaries for quality measuring
    cluster_cat_dict = (
        df_cluster.groupby(["cluster"])["category"]
        .apply(lambda x: [x for x in x])
        .to_dict()
    )
    cluster_len_dict = df_cluster["cluster"].value_counts().to_dict()

    # Prevents error when decreasing n_clusters and previous cluster_number of graph is higher
    try:
        cluster_cat_dict[cluster_number]
    except ValueError:
        cluster_number = 0

    # get quality score based on the correlation
    # between categories in the cluster
    # For clusters with only one item x_corr_mean is set to 0
    if len(cluster_cat_dict[cluster_number]) > 1:
        x_corr = data_pi_wide[cluster_cat_dict[cluster_number]].corr().abs()  # type: ignore
        # get the mean of the values in the upper triangle of the correlation matrix (and round to .2)
        x_corr_mean = round(
            x_corr.values[np.triu_indices_from(x_corr.values, 1)].mean(), 2
        )
    else:
        x_corr_mean = 0

    # plot all categories in one cluster
    if method == "TimeSeriesKMeans":
        plot_title = f"{method} cluster {cluster_number} (quality={x_corr_mean}, n={cluster_len_dict[cluster_number]}, metric={metric}, {nor})"
    else:
        plot_title = f"{method} cluster {cluster_number} (quality={x_corr_mean}, n={cluster_len_dict[cluster_number]}, {nor})"

    cols = cluster_cat_dict[cluster_number]
    fig = go.Figure()
    ind = data.index
    for _, col in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=ind,
                y=data[col],
                name=col,
                line={"width": 1},
                hoverlabel={"namelength": -1},
                showlegend=True,
                line_shape=line_shape,
            )
        )
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(title_text=plot_title, template="plotly_dark", height=600)
    fig.update_xaxes(title_text="months", dtick=[1, len(data.index) + 1])  # type: ignore
    return fig, n_clusters - 1
