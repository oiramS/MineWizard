from dash import Dash, dcc, html, Input, Output, State, callback,dash_table
import pandas as pd

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import io
import base64
import numpy as np
from .data_frame_transformer import Df_transformer
import plotly.express as px

df_transformer = Df_transformer()

def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            html.H1('Análisis Exploratorio de Datos'),
            #Explicacion de EDA
            html.Div(
            id="contenido",
            children=[
                html.P("El anáisis exploratorio de datos (Exploratory Data Análisis, EDA) es una serie de pasos para analizar sets de datos y extraer sus características principales."),
            ],            
         ),
            html.Div(
                id="upload-data",
                className="four columns",
                children=html.Div(
                    [
                        html.H4("Carga de dataset para Análisis Exploratorio de Datos", className="text-upload"),
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
                    html.Div(id='output-data-upload-EDA'),
                    ],
                ),
            )
        ] )
    )


def EDA(contents, filename):
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
        else:
            return html.Div([
            dbc.Alert('Hubo un error al cargar el archivo.', color="danger")
        ])
        df_transformer.set_dataframe(df)
    except Exception as e:
        print(e)
        return html.Div([
            dbc.Alert('Hubo un error al cargar el archivo.', color="danger")
        ])
    dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index().rename(columns={"index": "Column"})
    dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)  # Convertir los tipos de datos a strings

    nulls_df = pd.DataFrame(df.isnull().sum(), columns=["Null Count"]).reset_index().rename(columns={"index": "Column"})
    nulls_df['Null Count'] = nulls_df['Null Count'].astype(str)
    
    boxplot_graph = dcc.Graph(id='boxplot-graph')
    
    dropdown_options = [{'label': column, 'value': column} for column in df.select_dtypes(include=['int64', 'float64']).columns]

    dropdown = dcc.Dropdown(
        id='variable-dropdown',
        options=dropdown_options,
        value=dropdown_options[0]['value']
    )

    dropdown_boxplot = dcc.Dropdown(
        id='variable-dropdown-box',
        options=dropdown_options,
        value=dropdown_options[0]['value']
    )

    dataframe_store = dcc.Store(id='dataframe-store', data=df.to_dict('records'))
    histogram_graph = dcc.Graph(id='histogram-graph')

    # Obtener el resumen estadístico
    describe_df = df.describe().reset_index().rename(columns={"index": "Stat"})
    describe_df['Stat'] = describe_df['Stat'].astype(str)
    
    return html.Div([
        dbc.Alert(f'El archivo cargado es: {filename}', color="success"),
        # Se muestran las primeras 8 filas del dataframe
        html.Div(
            dash_table.DataTable(
                data=df.to_dict('records'),
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
        html.H5("Dimensiones del dataframe proporcionado:"),
        dbc.Row([
            dbc.Col([
                dbc.Alert("Número de filas: {}".format(df.shape[0]), color="info")
            ], width=3),  # Ajusta el ancho de la columna

            dbc.Col([
                dbc.Alert("Número de columnas: {}".format(df.shape[1]), color="info")
            ], width=3),
        ],
            justify='center'  # Añade la propiedad justify con el valor 'center'
        ),
        dbc.Row([
            dbc.Col([
                dbc.Alert(f'Variables numéricas: {df.select_dtypes(include="number").shape[1]}', color="info")
            ], width=3),  # Ajusta el ancho de la columna

            dbc.Col([
                dbc.Alert(f'Variables categóricas: {df.select_dtypes(exclude="number").shape[1]}', color="info")
            ], width=3),
        ],
            justify='center'  # Añade la propiedad justify con el valor 'center'
        ),
        html.H5("Descripción de los tipos de datos:"),
        create_table(dtypes_df),
        html.H5("Estadisticas del Dataset:"),
         html.Div(
            dash_table.DataTable(
                data=describe_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in describe_df.columns],
                fixed_columns={'headers': True, 'data': 1},
                style_cell={
                    'textAlign': 'left',
                    'padding': '1em',
                    'border': '1px solid black',
                    'borderRadius': '5px'
                },
                style_header={
                    'fontWeight': 'bold',
                    'border': '1px solid black',
                    'borderRadius': '5px'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Stat'},
                        'fontWeight': 'bold',
                    }
                ],
                style_table={
                    'height': 'auto',
                    'overflowX': 'auto',
                    'padding': 10,
                    'minWidth': '100%'
                },
            ),
            style={'marginLeft': 'auto',
                'marginRight': 'auto',
                'width':'90%'
                }
        ),
        html.H5("Descripción de los datos faltantes:"),
        create_table(nulls_df),
        html.H5("Diagramas de caja y bigotes para la detección de valores atípicos:"),
        html.Br(),
        html.Div([dropdown_boxplot, boxplot_graph]),
        dataframe_store,
        html.H5("Distribución de Variables Numéricas"),
        dcc.Dropdown(
            id='select_numeric_var',
            options=[{'label': value, 'value': value} for value in df.select_dtypes(include=np.number).columns.tolist()],
            value=None
            ),
        html.Div(id="hist-plots"),
        html.H5("Matriz de correlaciones"),
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
                            # font=dict(
                            #     color='white' if abs(df.corr().values[i][j]) >= 0.67  else 'black'
                            # ),
                        ) for i in range(len(df.corr(numeric_only=True).columns)) for j in range(i)
                    ],
                },
            },
        ),
    ],
)
@callback(Output('output-data-upload-EDA', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))

def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            EDA(c,n) for c,n in
            zip(list_of_contents, list_of_names)]
        return children

@callback(
    Output("hist-plots", "children"),
    Input("select_numeric_var", "value")
)
def create_hists(column_name):
    if column_name == None:
        return dbc.Alert('Selecciona una columna de la lista.', color="success"),
    df_to_show = df_transformer.get_df()
    return dcc.Graph(
        id='hist_plot',
        figure=px.histogram(df_to_show, x=column_name)
    )


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
    
def create_boxplot_figure(column, df):
    box = go.Box(
        x=df[column],
        name=column,
        marker=dict(color='rgb(0, 128, 128)'),
        boxmean=True
    )
    layout = go.Layout(
        title=f'Diagrama de caja y bigotes para {column}',
        yaxis=dict(title=column),
        xaxis=dict(title='Distribución'),
        hovermode='closest'
    )
    return go.Figure(data=[box], layout=layout)


@callback(
    Output('boxplot-graph', 'figure'),
    Input('variable-dropdown-box', 'value'),
    Input('dataframe-store', 'data')
)   
def update_boxplot(selected_variable, stored_data):
    df = pd.DataFrame(stored_data)
    figure = create_boxplot_figure(selected_variable, df)
    return figure
