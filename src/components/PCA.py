from dash import Dash, html
from dash import dcc, html, Input, Output, State, callback# Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
import pandas as pd

import dash_bootstrap_components as dbc
import io
from io import BytesIO
from dash import dash_table
import base64
import numpy as np
from dash.dash_table.Format import Group
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            
            html.H1('Análisis de Componentes Principales'),
            #Explicación de PCA
            html.Div(
            id="contenido",
            children=[
                html.P("El Análisis de Componentes Principales (en inglés Principal Component Analysis, PCA), es un método estadístico que nos permite reducir la dimensionalidad de los datos con los que estamos trabajando. Se utiliza cuando queremos elegir un menor número de predictores para pronosticar una variable objetivo, o para comprenderlos de una forma más simple (Morán, 2022)."),
            ],         
            ),
        #Sección para cargar archivo csv
        html.Div(
            
            id="upload-data",
                className="four columns",
                children=html.Div(
                    [
                        html.H4("Carga o elige el dataset para iniciar el Análisis Exploratorio de Datos", className="text-upload"),
                        # Muestra el módulo de carga
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                [
                                    'Haz click para seleccionar tu archivo csv',   
                                ],
                            ),
                        style={
                            'width': '50%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '2px',
                            'borderStyle': 'solid',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '2em auto',
                            'cursor': 'pointer',
                        },
                        multiple=True,
                        accept='.csv',
                        className="drag"
                        ),
                    html.Div(id='output-data-upload-pca'),
                    ],
                ),
                
            )
        ] 
    )
)
def estandarizacion(value,df):
    if value == 'StandardScaler':
        Estandarizar=StandardScaler()
    elif value == 'MinMaxScaler':
        Estandarizar=MinMaxScaler()
    else:
        print(value)
    return Estandarizar
def pricoman(Estandarizar,Corr):
    NuevaMatriz=Corr
    MEstandarizada=Estandarizar.fit_transform(NuevaMatriz)
    pd.DataFrame(MEstandarizada,columns=NuevaMatriz.columns)
    principal=PCA(n_components=None)
    principal.fit(MEstandarizada)
    print(principal.components_)
    return html.Div(
        dash_table.DataTable(
            data=MEstandarizada
        )
    )
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Se asume que el usuario cargó un archivo CSV 
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Se asume que el usuario cargó un archivo de excel
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            dbc.Alert('Hubo un error al cargar el archivo.', color="danger")
        ])

    return html.Div([
        dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),
        # Se muestran las primeras 8 filas del dataframe
        dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=8,
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=False,
            cell_selectable=True,
            editable=False,
            row_selectable='multi',
            columns=[{'name': i, 'id': i, "deletable":False} for i in df.columns],
            style_table={'height': '300px', 'overflowX': 'auto'},
        ),
        dbc.Alert('Variables numéricas: {}'.format(df.select_dtypes(include='number').shape[1]), color="info", class_name="my-3 mx-auto text-center w-25"),

        html.H3(
            "Evidencia de datos correlacionados"
        ),

        dcc.Graph(
            id='matriz',
            figure={
                'data': [
                    {'x': df.corr().columns, 'y': df.corr().columns, 'z': np.triu(df.corr().values, k=1), 'type': 'heatmap', 'colorscale': 'RdBu', 'symmetric': False}
                ],
                'layout': {
                    'title': 'Matriz de correlación',
                    'xaxis': {'side': 'down'},
                    'yaxis': {'side': 'left'},
                    # Agregamos el valor de correlación por en cada celda (text_auto = True)
                    'annotations': [
                        dict(
                            x=df.corr().columns[i],
                            y=df.corr().columns[j],
                            text=str(round(df.corr().values[i][j], 4)),
                            showarrow=False,
                            font=dict(
                                color='white' if abs(df.corr().values[i][j]) >= 0.67  else 'black'
                            ),
                        ) for i in range(len(df.corr().columns)) for j in range(i)
                    ],
                },
            },
        ),
        html.Div(
            children=[
                dbc.Badge(
                    "Identificación de correlaciones",
                    pill=True,
                    color="primary",
                    style={"font-size":"15px"}
                ),
                html.P("🟥 Correlación fuerte: De -1.0 a -0.67 y 0.67 a 1.0", className="ms-4"),
                html.P("⬜ Correlación moderada: De -0.66 a -0.34 y 0.34 a 0.66", className="ms-4"),
                html.P("🟦 Correlación débil: De -0.33 a 0.0 y 0.0 a 0.33", className="ms-4"),
                dbc.Alert("⚠️ Si no se identifica almenos una correlación fuerte, entonces PCA no aplica.", color="warning"),
            ],
            className="mt-3"
        ),
        html.H3(
            "Cálculo de Componentes Principales"
        ),
        html.Div(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ℹ️ Método de  Estandarización", color="primary",
                                            id="tooltip-method", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"},
                                            ),
                                            dbc.Tooltip(
                                                "Selecciona un método de estandarización.",
                                                target="tooltip-method"
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Select(
                                        id='select-escale',
                                        options=[
                                            {'label': 'StandardScaler', 'value': "StandardScaler"},
                                            {'label': 'MinMaxScaler', 'value': "MinMaxScaler"},
                                        ],
                                        value="StandardScaler",
                                        style={"font-size": "medium"},
                                        
    
                                    ),
                                   
                                    
                                    style={"height":"50px"},
                                ),
                               
                            ],
                            class_name="me-3"
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ℹ️ Núm. Componentes principales", color="primary",
                                                id="tooltip-numpc", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                "Elige la cantidad de componentes que quieras tomar en cuenta para el cálculo.",
                                                target="tooltip-numpc"
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='n_components',
                                        type='number',
                                        placeholder='None',
                                        value=None,
                                        min=1,
                                        max=df.select_dtypes(include='number').shape[1],
                                        style={"font-size": "medium"}
                                    ),
                                    style={"height":"50px"}
                                ),
                            ],
                            class_name="me-3"
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [dbc.Badge("ℹ️ Porcentaje de Relevancia", color="primary",
                                            id="tooltip-percentaje", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                        dbc.Tooltip("Muestra la cantidad de relevancia tomando en cuenta los conponentes seleccionados.",
                                                    target="tooltip-percentaje"
                                            ),
                                        ],
                                    style={"height":"50px"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='relevancia',
                                        type='number',
                                        placeholder='None',
                                        value=0.9,
                                        min=0.75,
                                        max=0.9,
                                        style={"font-size": "medium"}
                                    ),
                                ),
                            ],
                            class_name="me-3"
                        ),
                    ],
                    style={"justify-content": "between", "height": "100%"}
                ),
                

            ],
            style={"font-size":"20px"},
            className="mt-4",
        ),

    ]
)


@callback(Output('output-data-upload-pca', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))


def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c,n,d) for c,n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children



