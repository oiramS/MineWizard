from dash import Dash, html
from dash import dcc, html, Input, Output, State, callback, ALL
import pandas as pd
import dash_bootstrap_components as dbc
import io
from io import BytesIO
from dash import dash_table
import base64
import numpy as np
import dash
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from .data_frame_transformer import Df_transformer
import plotly.express as px

df_transformer = Df_transformer()

def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            html.H1('K Means'),
            #Explicación de Árboles de clasificación
            html.Div(
            id="contenido",
            children=[
                html.P("Esta técnica no supervisada se basa en identificar grupos en los datos de tal manera que todos los datos del grupo (clúster) son datos con características similares mientras que los datos de los otros grupos son diferentes."),
                ],         
            ),
            html.Div(
                id="upload-data",
                className="four columns",
                children=html.Div(
                    [
                        html.H4("Carga de dataset para iniciar K Means", className="text-upload"),
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
                    html.Div(id='output-data-upload-Class-kMean'),
                    ],
                ),
            )

        ] 
    )
 )

def kMeansFile(contents, filename, date):
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
         html.H5("Descripción general de los datos"),
         html.Div(
            create_data_table(df_transformer.get_info_as_df()),
        ),
        html.H5("Matriz de correlaciones:"),
        dcc.Graph(
            id='matriz',
            figure={
                'data': [
                    {'x': df.corr(numeric_only=True).columns, 'y': df.corr().columns, 'z': np.triu(df.corr().values, k=1), 'type': 'heatmap', 'colorscale': 'sepal_length', 'color_continuous_scale':'scale' , 'symmetric': False}
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
        html.H3("Selección de variables"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Selecciona las variables a considerar:"),
                    dcc.Dropdown(id="feature-columns-dropdown-kMean", multi=True)
                ], 
                style={"margin-bottom": "20px", "padding":20}
                ),
            ]),
            

        ]),
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
        html.Div(id='estandar-kMeans'),
        html.H3("Selecciona los parámetros para K Means"),
        html.Div([
            dbc.Row([
            dcc.Input(id="n_clusters", type="number", value=None,placeholder="Número de clústers"),
            ],
            style={
                'marginTop' : '10px', 
            }),  
            dbc.Row([
            dcc.Input(id="n_init", type="number",value=None,placeholder="Número de iteraciones para clústers"),
           ],
            style={
                'marginTop' : '10px', 
            }),   
            
            html.Button(
                "Generar Modelo", 
                id="generate-km-button", 
                n_clicks=0,
                className="btn btn-success",
                style={
                    'marginTop' : '10px',
                    'marginLeft': '75%',
                }
            ),
            ],     
            style={
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'width': '80%',
                'padding':10,
            }        
        ),
        
        html.Div(id="prediction-output-kMean"),
            
              
    ],
    style={
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'width': '90%',
                'padding':10,
            }
)
@callback(Output('output-data-upload-Class-kMean', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            kMeansFile(c,n,d) for c,n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

@callback(
    Output("feature-columns-dropdown-kMean", "options"),
    Input("feature-columns-dropdown-kMean", "value")
)
def update_column_options(target_column):
    # Read the data from a CSV file (assuming it's named "data.csv")
    data = df_transformer.get_df()
    data = data.select_dtypes(['number'])
    # Get the column names from the data
    columns = data.columns.tolist()
    
    # Create the options for the dropdown menus
    options = [{"label": col, "value": col} for col in columns]   
    return options

@callback(
    Output("prediction-output-kMean", "children"),
    Input("generate-km-button", "n_clicks"),
    State("n_clusters", "value"),
    State("n_init", "value"),
    State("feature-columns-dropdown-kMean","value"),
)
def generate_model(n_clicks,clusters,init,feature_columns):
    if(feature_columns != None and n_clicks>0):
        if clusters == None:
            return  html.Div([
                dbc.Alert('Debes ingresar un número clusters', color="danger")
            ])
        
        MEstandarizada = df_transformer.get_cur_mscaler()
        
        # Initialize the Decision Tree Classifier
        km= KMeans( n_clusters=clusters,random_state=0, n_init=10).fit(MEstandarizada)
        km.predict(MEstandarizada)
        km_result_df = df_transformer.get_df()[feature_columns]
        km_result_df['clusterP'] = km.labels_
        df_transformer.set_kmdf(km_result_df)
        cluster_count = km_result_df.groupby(['clusterP'])['clusterP'].count().to_frame()
        cluster_count.insert(loc=0, column='cluster', value=[i for i in range(clusters)])
        centroides = km_result_df.groupby('clusterP').mean()
        centroides.insert(loc=0, column='cluster', value=[i for i in range(clusters)])
        return html.Div([
            html.H5("Resultado de la clasificación"),
            html.Div(
                create_data_table(km_result_df)
                ),
            html.H5("Cantidad de elementos en los clusters"),
            html.Div(
                create_data_table(cluster_count)
                ),
            html.H5("Centroides"),
            html.Div(
                create_data_table(centroides)
                ),
            html.H5("Visualizar datos por cluster"),
            dcc.Dropdown(
                id="select-kMean-cluster", 
                options=[{'label': f'cluster {i}', 'value': i} for i in range(clusters)],
                value=0
                ),
            html.Div(id='filter_by_cluster'),
            
            ],
            style={
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'width': '90%',
                'padding':10,
            } 
        )
        
        
@callback(
    Output("filter_by_cluster", "children"),
    Input("select-kMean-cluster", "value")
)
def create_inputs(cluster_id):
    df_to_show = df_transformer.get_kmdf()
    rslt = df_to_show.loc[df_to_show['clusterP'] == cluster_id]
    return html.Div(
        create_data_table(rslt)
    )


      

@callback(
    Output("feature-inputs-div-kMean", "children"),
    Input("feature-columns-dropdown-kMean", "value")
)
def create_inputs(inputs):
    input_elements = []
    for index, inp in enumerate(inputs):
        input_elements.append(
            dbc.Row([
            dcc.Input(id={"type":"feature-input-value", "index":index}, type="number",  placeholder=inp)
            ],
            style={
                'marginTop' : '10px',
            })
        )
    return input_elements

def create_data_table(estandarizado)->html.Table:
    return html.Table(
            dash_table.DataTable(
                data=estandarizado.to_dict('records'),
                page_size=10,
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
                    #'height': '300px', 
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
                'width': '100%'
                
                }
        )

@callback(
    Output(component_id='estandar-kMeans', component_property='children'),
    Input(component_id='select-escale', component_property='value'),
    Input(component_id="feature-columns-dropdown-kMean", component_property="value")
)
def update_estandar(value,featureColumns):
    if featureColumns == None:
        return ''
    df = df_transformer.get_df()[featureColumns]
    if value == "MinMaxScaler":
        estandarizado=MinMaxScaler().fit_transform(df)
    else:
        estandarizado=StandardScaler().fit_transform(df)
    
    df_estandarizado = pd.DataFrame(np.around(estandarizado,5),columns=df.columns)
    df_transformer.set_cur_scaler(df_estandarizado)
    df_transformer.set_cur_mscaler(estandarizado)
    sse = []
    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=0, n_init=10)
        km.fit(estandarizado)
        sse.append(km.inertia_)
    
    df_for_plot = pd.DataFrame(sse, columns=['SSE'], index=[i for i in range(2,10)])
    if featureColumns != None:
        return html.Div([
            html.H5("Matriz Estandarizada"),
            html.Div(
                create_data_table(df_estandarizado[featureColumns]),
            ),
            dcc.Graph(
                id='elbow-plot',
                figure=px.line(df_for_plot, 
                            #    x='Cantidad de clusters k', 
                            #    y='SSE',
                               title='Elbow Method',
                               markers=True)
            )
        ])
    return ""