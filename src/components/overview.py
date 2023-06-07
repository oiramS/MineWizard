from dash import Dash, html


def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    my_list=['''MineWizard permite a los usuarios importar datos de tipo CSV.''',
                '''Proporciona capacidades de preprocesamiento para limpiar y transformar 
                datos sin procesar, incluido el manejo de valores faltantes, detección de valores atípicos y escalado de variables.''',
                 '''Ofrece visualizaciones interactivas, resúmenes estadísticos y herramientas de creación de perfiles de
                   datos para comprender la distribución, la correlación y las características clave de los datos.''',''' 
                   Integra una amplia gama de algoritmos de minería de datos, que incluyen clasificación, regresión, agrupamiento (clústering).''',
                '''Admite el entrenamiento de modelos utilizando varios algoritmos y ofrece opciones personalizables para el ajuste de parámetros.''',
                '''Los usuarios pueden evaluar el rendimiento de los modelos entrenados utilizando métricas como exactitud y precisión.'''                  ]
    return html.Div(
        children=html.Div([
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '25em'},
                children=[
                    html.Img(
                        id="MinewizardLogo",
                        src="/assets/images/logo.PNG",
                        style = {'width': '20em'}
                    )
                ]
            ),
            html.H1('Descripción general'),
            html.Div(
                children=[
                html.P('''MineWizard es una herramienta creada para aprovechar algunas técnicas 
                de minería de datos para extraer información valiosa y patrones 
                de grandes conjuntos de datos. Permite a los usuarios interactuar 
                con los datos, realizar análisis y descubrir relaciones, tendencias
                y patrones ocultos. Aprovecha técnicas avanzadas de minería de datos, 
                algoritmos de aprendizaje automático e interfaces de usuario intuitivas 
                para facilitar la exploración, el análisis y la toma de decisiones de datos.
            '''  ),
            
                ],style={'display': 'flex', 'align-items': 'center', 'justify-content': 'justify','text-align':'justify' }),
            html.H4('Características clave:'),
            html.Div(
                id="contenido",
                children=[
                
                    html.Ul([html.Li(x) for x in my_list]
                            
                            ),
                    
            
                    ],style={'display': 'flex', 'align-items': 'center', 'justify-content': 'justify','text-align':'justify' }),
            

        ] ),
        style={
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'width': '90%',
                'padding':10,
        } 
    )