from dash import Dash, dcc, html, Input, Output, callback
from pages import overview, clustering_pi, clustering_cr, clustering_lo, method_quality
#import dash_bootstrap_components as dbc

app = Dash(__name__, suppress_callback_exceptions=True)#, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
app.title = "Seasonality Clustering"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('| Overview |', href='/'),
        dcc.Link(' Page Impressions |', href='/pages/clustering_pi'),
        dcc.Link(' Conversion Rate |', href='/pages/clustering_cr'),
        dcc.Link(' Lead-out |', href='/pages/clustering_lo'),
        dcc.Link(' Quality of methods |', href='/pages/method_quality'),
    ], className="row"),
    html.Div(id='page-content', children=[])
])

@callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return overview.layout
    elif pathname == '/pages/clustering_pi':
        return clustering_pi.layout
    elif pathname == '/pages/clustering_cr':
        return clustering_cr.layout
    elif pathname == '/pages/clustering_lo':
        return clustering_lo.layout
    elif pathname == '/pages/method_quality':
        return method_quality.layout    
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=False)

