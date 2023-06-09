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
import plotly.express as px
import plotly.graph_objects as go

from .data_frame_transformer import Df_transformer

df_transformer = Df_transformer()

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
                html.P("El Análisis de Componentes Principales (en inglés Principal Component Analysis, PCA), es un método estadístico que nos permite reducir la dimensionalidad de los datos con los que estamos trabajando. Se utiliza cuando queremos elegir un menor número de predictores para pronosticar una variable objetivo, o para comprenderlos de una forma más simple."),
            ],         
            ),
        #Sección para cargar archivo csv
        html.Div(
            
            id="upload-data",
                className="four columns",
                children=html.Div(
                    [
                        html.H4("Carga dataset para Análisis de Componentes Principales", className="text-upload"),
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
        df_transformer.set_dataframe(df)
    except Exception as e:
        print(e)
        return html.Div([
            dbc.Alert('Hubo un error al cargar el archivo.', color="danger")
        ])

    return html.Div([
        dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),
        # Se muestran las primeras 8 filas del dataframe
        html.Div(
            create_data_table(df_transformer.get_df()),
        ),
        
        dbc.Alert('Cantidad de variables totales: {}'.format(df.select_dtypes(include='number').shape[1]), color="info", class_name="my-3 mx-auto text-center w-25"),

        html.H3(
            "Correlaciones de datos"
        ),

        dcc.Graph(
            id='matriz',
            figure={
                'data': [
                    {'x': df.corr(numeric_only=True).columns, 'y': df.corr(numeric_only=True).columns, 'z': np.triu(df.corr(numeric_only=True).values, k=1), 'type': 'heatmap', 'colorscale': 'sepal_length', 'color_continuous_scale':'scale' , 'symmetric': False}
                ],
                'layout': {
                    'title': 'Matriz de correlación',
                    'xaxis': {'side': 'down'},
                    'yaxis': {'side': 'left'},
                    # 'plot_bgcolor':'rgba(0,0,0,0)', 
                    # 'paper_bgcolor':'rgba(0,0,0,0)',
                    # 'font' : {'color' : '#7FDBFF'},
                    # Agregamos el valor de correlación por en cada celda (text_auto = True)
                    'annotations': [
                        dict(
                            x=df.corr(numeric_only=True).columns[i],
                            y=df.corr(numeric_only=True).columns[j],
                            text=str(round(df.corr(numeric_only=True).values[i][j], 4)),
                            showarrow=False,
                            font=dict(
                                color='white' #if abs(df.corr(numeric_only=True).values[i][j]) >= 0.67  else 'black'
                            ),
                        ) for i in range(len(df.corr(numeric_only=True).columns)) for j in range(i)
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
                html.P("Correlación positiva fuerte: De -1.0 a -0.67 y 0.67 a 1.0", className="ms-4"),
                html.P("Correlación débil: De -0.66 a -0.34 y 0.34 a 0.66", className="ms-4"),
                html.P("Correlación negativa fuerte: De -0.33 a 0.0 y 0.0 a 0.33", className="ms-4"),
                dbc.Alert("Si no se identifica almenos una correlación fuerte, entonces PCA no aplica.", color="warning"),
            ],
            className="mt-3"
        ),
        html.H3(
            "Cálculo de Componentes Principales"
        ),
        html.Div(

            children=[
                    dbc.Badge("Método de  Estandarización", color="primary",
                    id="tooltip-method", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"},
                    ),
                    dbc.Tooltip(
                        "Selecciona un método de estandarización.",
                        target="tooltip-method"
                    ),
                    dbc.Select(
                        
                    id='select-escale',
                    options=[
                        {'label': 'Estándar', 'value': "StandardScaler"},
                        {'label': 'MinMax', 'value': "MinMaxScaler"},
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
            dbc.Badge("Cantidad de Componentes principales", color="primary",
                id="tooltip-numpc", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
            ),
            dbc.Tooltip(
                        "Introduce el número de componentes principales.",
                        target="tooltip-numpc"
                    ),
            dbc.Input(
                id='n_components',
                type='number',
                placeholder='5',
                value=None,
                min=1,
                max=df.select_dtypes(include='number').shape[1],
                style={"font-size": "medium"}
            ),


            ],
            style={"font-size":"20px"},
            className="mt-4",
        ),
        html.Div(id='numComp'),
        html.Div(id='variance'),
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
    Input(component_id='select-escale', component_property='value'),
)
def update_estandar(value):
    df = df_transformer.get_df()
    df = df[df.select_dtypes(include=np.number).columns.tolist()]
    if value == "MinMaxScaler":
        estandarizado=MinMaxScaler().fit_transform(df)
    else:
        estandarizado=StandardScaler().fit_transform(df)
    df_estandarizado = pd.DataFrame(np.around(estandarizado,5),columns=df.columns)
    df_transformer.set_cur_scaler(df_estandarizado)

    return html.Div(
        create_data_table(df_estandarizado)
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
                columns=[{'name': i, 'id': i, "deletable":False} for i in estandarizado.columns],
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
                'width': '90%'
                
                }
        )
    
@callback(
    Output(component_id='numComp', component_property='children'),
    Input(component_id='n_components', component_property='value'),
    Input(component_id='select-escale', component_property='value'))
def update_components(value, state):
    if value != None:
        df = df_transformer.get_df()
        df.select_dtypes(include=np.number).columns.tolist()
        pca=PCA(n_components=value)
        pca.fit(df_transformer.get_cur_scaler())
        components=pd.DataFrame(abs(np.around(pca.components_, 5)),columns=df.select_dtypes(include=np.number).columns)
        varianza=pca.explained_variance_ratio_
        df_transformer.set_varianza(varianza)
        #print(pca.explained_variance_ratio_)
        sum_cum = np.cumsum(pca.explained_variance_ratio_)
        #sum_cum.insert(0,0)
        fig = go.Figure(data=go.Scatter(x=[i+1 for i in range(len(sum_cum))], y = sum_cum))
        fig.update_layout(title='Gráfica de la Varianza Acumulada',
                   xaxis_title='Número de Componentes',
                   yaxis_title='Varianza Acumulada')
        return html.Div([
            html.Div(create_data_table(components)),
            dbc.Alert(
                f"Proporción de varianza: {varianza}"
            ),
            dbc.Alert(
                f"Varianza acumulada para: {value} componentes: {round(df_transformer.get_varianza_acum(value),3)}"
            ),
            dcc.Graph(
                figure = fig
            ),
            html.H5("Seleccione los componentes a eliminar"),
            dcc.Dropdown(
                id='select_principal_components',
                options=[{'label': value, 'value': value} for value in df_transformer.get_df().select_dtypes(include=np.number).columns.tolist()],
                value=None,
                multi=True   
            ),
            html.Div(id="pca_df"),
            
            ])
    return html.Div(
        [
        dbc.Alert(
            "Esperando número de componentes principales..."
        )
        ],
    style={
        'marginLeft': 'auto',
        'marginRight': 'auto',
        'width': '80%',
        'padding':10,
    })
        
@callback(
    Output(component_id='pca_df', component_property='children'),
    Input(component_id="select_principal_components", component_property="value")
)
def update_estandar(featureColumns):
    if featureColumns == None or not any(featureColumns):
        df_pca = df_transformer.get_df()
    else:
        df_pca = df_transformer.get_df().drop(columns=featureColumns)
    return html.Div([
        html.Div(create_data_table(df_pca)),
        html.H5("Análisis Correlacional de Datos"),
        dbc.Row([
            html.H6("Columna en el eje X"),
            dcc.Dropdown(
                id='x_value',
                options=[{'label': value, 'value': value} for value in df_pca.select_dtypes(include=np.number).columns.tolist()],
                value=None,
                multi=False   
            ),
            ],
            style={
                'marginTop' : '10px', 
            }),  
            dbc.Row([
            html.H6("Columna en el eje Y"), 
            dcc.Dropdown(
                id='y_value',
                options=[{'label': value, 'value': value} for value in df_pca.select_dtypes(include=np.number).columns.tolist()],
                value=None,
                multi=False   
            ),
           ],
            style={
                'marginTop' : '10px', 
            }),   
            dbc.Row([
            html.H6("Columna de agrupación"),
            dcc.Dropdown(
                id='hue_value',
                options=[{'label': value, 'value': value} for value in df_pca.columns.tolist()],
                value=None,
                multi=False   
            ),
            ],
            style={
                'marginTop' : '10px', 
            }),
            html.Div(id='acd_scatter')
    ],
    style={
        'marginLeft': 'auto',
        'marginRight': 'auto',
        'width': '80%',
        'padding':10,
    }     
    )
    
@callback(
Output(component_id='acd_scatter', component_property='children'),
Input(component_id="x_value", component_property="value"),
Input(component_id="y_value", component_property="value"),
Input(component_id="hue_value", component_property="value")
)
def build_scatter_plot(x, y, hue):
    if x == None or y == None or hue == None:
        return dbc.Alert("Complete el formulario", color="success"),
    df = df_transformer.get_df()
    return dcc.Graph(
                id='elbow-plot',
                figure=px.scatter(df, 
                               x=x, 
                               y=y,
                               color= hue,
                               title='Gráfico de dispersión',
                            )
            )