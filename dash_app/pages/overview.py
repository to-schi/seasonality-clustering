from dash import dcc, html, Input, Output, callback
import pandas as pd
import pathlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
data = pd.read_csv(DATA_PATH.joinpath("data.csv"), sep=";")

category_list = sorted(data["category"].unique().tolist())
all_cat_mean = ["All categories (mean)"]
category_list = all_cat_mean + category_list
# upload new data? https://dash.plotly.com/dash-core-components/upload

def monthly_mean_chart():
    monthly_pi = data['pageimpressions'].groupby(data['month']).mean()
    monthly_cr = data['CR'].groupby(data['month']).mean()
    data["lead-out"] = round(data['pageimpressions'] * data['CR'])
    monthly_lo = data['lead-out'].groupby(data['month']).mean()

    fig = make_subplots(specs=[[{"secondary_y": True, "type": "xy"}]])
    fig.add_trace(go.Scatter(y=monthly_pi, x=monthly_pi.index, name="page impressions"), secondary_y=False)
    fig.add_trace(go.Scatter(y=monthly_cr*100, x=monthly_cr.index, name="conversion rate"), secondary_y=True)
    fig.add_trace(go.Scatter(y=monthly_lo, x=monthly_lo.index, name="lead-out"), secondary_y=False)
    
    fig.update_layout(title_text="Page impressions, conversion rate & lead-out (all categories, mean)", template='plotly_dark', height=600)
    fig.update_xaxes(title_text="months", dtick=[1,len(monthly_cr.index)])
    fig.update_yaxes(title_text="page impressions / lead-out", secondary_y=False, range=[10000,90000])
    fig.update_yaxes(title_text="conversion rate %", secondary_y=True, range=[24,26])
    return fig

layout = html.Div([
    html.H2("Dashboard for the Analysis of Seasonality Patterns in 100 e-Commerce Product Categories"),
    html.H3('Page Impressions and Conversion Rate by Category (2019'),
    html.H6('Choose a category:'),
    dcc.Dropdown(
        id="dropdown",
        options=category_list,
        value=category_list[0],
        clearable=False,
    ),
    dcc.Graph(id="graph1"),
])

@callback(
    Output(component_id="graph1", component_property="figure"), 
    Input(component_id="dropdown", component_property="value"),
    )
def update_bar_chart(category):
    if category == "All categories (mean)":
        fig = monthly_mean_chart()
    else:
        cat = data.loc[data['category'] == category]
        fig = go.Figure(
        data=[
            go.Bar(name='page impressions', x=cat["month"], y=cat["pageimpressions"], yaxis='y', offsetgroup=1),
            go.Bar(name='conversion rate', x=cat["month"], y=cat["CR"]*100, yaxis='y2', offsetgroup=2)
        ],
        layout={
            'yaxis': {'title': 'page impressions'},
            'yaxis2': {'title': 'conversion rate %', 'overlaying': 'y', 'side': 'right'}
        }
        )
        fig.update_xaxes(title_text="months", dtick=[len(cat.index)])
        fig.update_layout(title_text=f"{category}", 
                    barmode='group', template='plotly_dark', height=600,
                    legend=dict(yanchor="bottom",y=0.99, xanchor="right",x=0.99))
    return fig

