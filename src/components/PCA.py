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
    global Estandarizar
    if value == 'StandardScaler':
        Estandarizar=StandardScaler().fit_transform(df)
    elif value == 'MinMaxScaler':
        Estandarizar=MinMaxScaler().fit_transform(df)
    return pd.DataFrame(Estandarizar,columns=df.columns)



def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    global df
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
        html.Div(
            create_data_table(df),
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
                html.P("🟥 Correlación positiva fuerte: De -1.0 a -0.67 y 0.67 a 1.0", className="ms-4"),
                html.P("⬜ Correlación débil: De -0.66 a -0.34 y 0.34 a 0.66", className="ms-4"),
                html.P("🟦 Correlación negativa fuerte: De -0.33 a 0.0 y 0.0 a 0.33", className="ms-4"),
                dbc.Alert("⚠️ Si no se identifica almenos una correlación fuerte, entonces PCA no aplica.", color="warning"),
            ],
            className="mt-3"
        ),
        html.H3(
            "Cálculo de Componentes Principales"
        ),
        html.Div(

            children=[
                    dbc.Badge("ℹ️ Método de  Estandarización", color="primary",
                    id="tooltip-method", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"},
                    ),
                    dbc.Tooltip(
                        "Selecciona un método de estandarización.",
                        target="tooltip-method"
                    ),
                    dbc.Select(
                        
                    id='select-escale',
                    options=[
                        {'label': 'StandardScaler', 'value': "StandardScaler"},
                        {'label': 'MinMaxScaler', 'value': "MinMaxScaler"},
                    ],
                    value="StandardScaler",
                    style={"font-size": "medium"},
                                        
    
                                    ),
            ],
            style={"font-size":"20px"},
            className="mt-4",
        ),
        html.Div(id='estandar'),
        html.Div(
            children=[
            dbc.Badge("ℹ️ Núm. Componentes principales", color="primary",
                id="tooltip-numpc", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
            ),
            dbc.Tooltip(
                "Elige la cantidad de componentes que quieras tomar en cuenta para el cálculo.",
                target="tooltip-numpc"
            ),
            dbc.Input(
                id='n_components',
                type='number',
                placeholder='Ej:5',
                value=3,
                min=1,
                max=df.select_dtypes(include='number').shape[1],
                style={"font-size": "medium"}
            ),


            ]
        ),
        html.Div(id='numComp'),
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


@callback(
    Output(component_id='estandar', component_property='children'),
    Input(component_id='n_components', component_property='value'),
    State(component_id='select-escale', component_property='value')
)
def update_estandar(n_components,value):
    global estandarizado
    estandarizado=estandarizacion(value,df)

    return html.Div(
        create_data_table(estandarizado)
    )
def create_data_table(estandarizado)->html.Table:
    return html.Table(
            dash_table.DataTable(
                data=estandarizado.to_dict('records'),
                page_size=8,
                sort_action='native',
                sort_mode='multi',
                column_selectable='single',
                row_deletable=False,
                cell_selectable=False,
                editable=False,
                row_selectable='multi',
                columns=[{'name': i, 'id': i, "deletable":False} for i in df.columns],
                style_table={
                    'padding': 10,
                    'height': '300px', 
                    'overflowX': 'auto',
                    'minWidth': '100%'
                    },
            ),
            style={'marginLeft': 'auto',
                'marginRight': 'auto',
                'width':'90%'
                }
        ),



def create_table(datatypes) -> html.Table:
    return html.Table(

            dash_table.DataTable(
            data=datatypes.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in datatypes.columns],
            fixed_rows={'headers': True},
                style_cell={
                    'textAlign': 'left',
                    'padding': '1em',
                    'whiteSpace': 'normal',
                    'width': '50%',
                    
                },
                style_header={
                    'fontWeight': 'bold',
                },
                style_table={
                    'height': '600px',
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                }
            ),
            style={
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'width': 500
                
                }
        )
    
@callback(
    Output(component_id='numComp', component_property='children'),
    Input(component_id='n_components', component_property='value'))
def update_components(value):
    pca=PCA(n_components=value)
    pca.fit(Estandarizar)
    components=pd.DataFrame(abs(pca.components_),columns=df.columns)
    varianza=pca.explained_variance_ratio_
    varianza_acumulada= sum(varianza[0:value])
    #return create_data_table(components)
    return html.Div(
        [
        create_table(components),
        html.Div(
            f"Proporción de varianza: {varianza}"
        ),
        html.Div(
            f"Varianza acumulada para: {value} componentes: {varianza_acumulada}"
        )
        ])

