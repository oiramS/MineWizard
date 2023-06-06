from dash import Dash, html
from dash import dcc, html, Input, Output, State, callback# Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
import pandas as pd

import dash_bootstrap_components as dbc
import io
from io import BytesIO
from dash import dash_table
import base64
import numpy as np

def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            html.H1('Bosque de pronóstico'),
            #Explicación de Árboles de pronóstico
            html.Div(
            id="contenido",
            children=[
                html.P("Un bosque aleatorio es un grupo de árboles de decisión.elegirá características al azar y hará observaciones, construirá un bosque de árboles de decisión y luego promediará los resultados.  Es uno de los algoritmos más utilizados debido a su precisión, simplicidad y flexibilidad."),
            
                ],         
            ),
            html.Div(
                id="upload-data",
                className="four columns",
                children=html.Div(
                    [
                        html.H4("Carga de dataset para iniciar el Bosque de pronóstico", className="text-upload"),
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
                    html.Div(id='output-data-upload-ProFor'),
                    ],
                ),
            )

        ] 
    )
 )

def Prontree(contents, filename, date):
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
    ],
)
@callback(Output('output-data-upload-ProFor', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            Prontree(c,n,d) for c,n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children
