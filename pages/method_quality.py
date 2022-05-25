from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objects as go
import pathlib

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
method_quality = pd.read_csv(DATA_PATH.joinpath("method_quality.csv"), index_col='n_clusters', sep=";")
method_quality_norm = pd.read_csv(DATA_PATH.joinpath("method_quality_norm.csv"), index_col='n_clusters', sep=";")

layout = html.Div([
    html.H3('Measuring the quality of clustering methods'),
    dcc.Dropdown(
        id="dropdown_methods",
        options=[
            {'label': 'Not normalized "page impressions" data', 'value': 'not normalized'},
            {'label': 'Normalized "page impressions" data', 'value': 'normalized'},
        ],
        value='not normalized',
        clearable=False,        
    ),
    dcc.Graph(id="graph_methods"),
])

@callback(
    Output(component_id='graph_methods', component_property='figure'),
    Input(component_id='dropdown_methods', component_property='value')
)
def update_chart(norm):
    if norm == 'normalized':
        df = method_quality_norm
    else: 
        df = method_quality

    fig = go.Figure()
    plot_title = f"Clustering quality measurement with silhouette score ({norm} data)"
    # Loop df columns and plot columns to the figure
    for i in range(0, len(df.columns)):
        col_name = df.columns.values[i]
        fig.add_trace(go.Scatter(x=df.index, y=df[col_name], mode='lines', name=col_name))
    fig.update_xaxes(title_text="n_clusters", tickmode='linear', tick0=1, dtick=1)
    fig.update_yaxes(title_text="silhouette score")
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(title_text=plot_title, template="plotly_dark", height=600)
    return fig

