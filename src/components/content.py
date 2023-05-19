from dash import Dash, html, Input, Output
from . import overview, EDA, PCA, PronForest, PronTree, notFound

def render(app: Dash) -> html.Div:
    '''
    navbar 
    '''
    content = html.Div(id="page-content")
    @app.callback(
        Output("page-content", "children"),
        [Input("url", "pathname")]
        )
    def display_module(pathname) -> html.Div:
        if pathname == "/":
            return overview.render(app)
        elif pathname == "/eda":
            return EDA.render(app)
        elif pathname == "/pca":
            return PCA.render(app)
        elif pathname == "/c_tree":
            return notFound.render(app)
        elif pathname == "/c_forest":
            return notFound.render(app)
        elif pathname == "/p_tree":
            return PronTree.render(app)
        elif pathname == "/p_forest":
            return PronForest.render(app)
        else:
            return notFound.render(app)
    return content