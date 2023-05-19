from dash import Dash, html, dcc


from . import navbar, content


def create_layout(app: Dash) -> html.Div:
    '''
    layout entrypoint, here we render all the site components
    '''
    return html.Div([dcc.Location(id = "url"),
    navbar.render(app),
    content.render(app)
])
