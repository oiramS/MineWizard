from dash import Dash, html

app = Dash(__name__)

app.layout = html.Div([
    html.Div(
        className="app-header",
        children=[
            html.Div('MineWizard', className="app-header--title")
        ]
    ),
    html.Div(
        children=html.Div([
            html.H1('Overview'),
            html.Div('''

                This is an example of a simple Dash app with
                local, customized CSS.
            '''    ,className="example"    )

        ] )
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)