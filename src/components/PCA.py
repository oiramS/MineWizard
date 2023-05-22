from dash import Dash, html


def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            html.H1('Análisis de Componentes Principales'),
            html.P('''
                
                El Análisis de Componentes Principales (en inglés Principal Component Analysis, PCA), 
                es un método estadístico que nos permite reducir la dimensionalidad de los datos con los
                que estamos trabajando. Se utiliza cuando queremos elegir un menor número de predictores 
                para pronosticar una variable objetivo, o para comprenderlos de una forma más simple (Morán, 2022).
            '''),
            html.Div('''
                
                El Análisis de Componentes Principales (en inglés Principal Component Analysis, PCA), 
                es un método estadístico que nos permite reducir la dimensionalidad de los datos con los
                que estamos trabajando. Se utiliza cuando queremos elegir un menor número de predictores 
                para pronosticar una variable objetivo, o para comprenderlos de una forma más simple.
            '''    ,className="example")

        ] )
    )
    