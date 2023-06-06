from dash import Dash, html


def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            html.H1('Descripción general'),
            html.Div('''
                MineWizard es una herramienta creada para aprovechar algunas técnicas 
                de minería de datos para extraer información valiosa y patrones 
                de grandes conjuntos de datos. Permite a los usuarios interactuar 
                con los datos, realizar análisis y descubrir relaciones, tendencias
                y patrones ocultos.
            '''    ,className="example")

        ] ),
        style={
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'width': '90%',
                'padding':10,
        } 
    )