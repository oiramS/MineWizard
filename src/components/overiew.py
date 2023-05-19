from dash import Dash, html


def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            html.H1('Overview'),
            html.Div('''

                This is an example of a simple Dash app with
                local, customized CSS.
            '''    ,className="example")

        ] )
    )