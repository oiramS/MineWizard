from dash import Dash, html


def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            html.H1('Análisis Exploratorio de Datos'),
            html.P('''El anáisis exploratorio de datos (Exploratory Data Análisis, EDA) es
                una serie de pasos para analizar sets de datos 
                y extraer sus características principales (Granados-López, 2021)'''),
            html.Div('''

                This is an example of a simple Dash app with
                local, customized CSS.
            '''    ,className="example")

        ] )
    )
    