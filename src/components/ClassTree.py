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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.express as px
from sklearn.tree import export_text
from .data_frame_transformer import Df_transformer

df_transformer = Df_transformer()

def render(app: Dash) -> html.Div:
    '''
    overview
    '''
    return html.Div(
        children=html.Div([
            html.H1('Árbol de clasificación'),
            #Explicación de Árboles de clasificación
            html.Div(
            id="contenido",
            children=[
                html.P("El Árbol de clasificación, cuando la variable dependiente es de tipo cualitativa. Se trata de dar con un esquema de múltiples bifuraciones, anidadas en forma de árbol, para obtener una predicción para la clase de pertenencia de un elemento."),
            
                ],         
            ),
            html.Div(
                id="upload-data",
                className="four columns",
                children=html.Div(
                    [
                        html.H4("Carga de dataset para iniciar el Árbol de clasificación", className="text-upload"),
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
                    html.Div(id='output-data-upload-ClassTree'),
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
                    html.Label("Selecciona las variables predictoras:"),
                    dcc.Dropdown(id="feature-columns-dropdown-cTree", multi=True)
                ], 
                style={"margin-bottom": "20px", "padding":20}
                ),
            ]),
            dbc.Col([
                html.Div([
                    html.Label("Selecciona la variable a clasificar:"),
                    dcc.Dropdown(id="target-column-dropdown-cTree")
                ], 
                style={"margin-bottom": "20px", "padding": 20}
                ),
            ]),

        ]),
        html.H3("Selecciona los parámetros para el árbol"),
        html.Div([
            dbc.Row([
            dcc.Input(id="max_depth", type="number", value=None,placeholder="Profundidad del árbol"),
            ],
            style={
                'marginTop' : '10px', 
            }),  
            dbc.Row([
            dcc.Input(id="min_samples_split", type="number",value=None,placeholder="Mínimo de muestras para dividir"),
           ],
            style={
                'marginTop' : '10px', 
            }),   
            dbc.Row([
            dcc.Input(id="min_samples_leaf", type="number",value=None,placeholder="Mínimo de muestras en hojas")
            ],
            style={
                'marginTop' : '10px', 
            }),
            html.Button(
                "Generar Modelo", 
                id="generate-button", 
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
        
        html.Div(id="prediction-output-cTree"),
            
              
    ],
    style={
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'width': '90%',
                'padding':10,
            }
)
@callback(Output('output-data-upload-ClassTree', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            Prontree(c,n,d) for c,n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

@callback(
    Output("target-column-dropdown-cTree", "options"),
    Output("feature-columns-dropdown-cTree", "options"),
    Input("target-column-dropdown-cTree", "value"),
    State("feature-columns-dropdown-cTree", "value")
)
def update_column_options(target_column, feature_columns):
    # Read the data from a CSV file (assuming it's named "data.csv")
    data = df_transformer.get_df()
    data = data.select_dtypes(['number'])
    # Get the column names from the data
    columns = data.columns.tolist()
    
    # Create the options for the dropdown menus
    options = [{"label": col, "value": col} for col in columns]
    option = [{"label": col, "value": col} for col in columns]
    
    return option, options

@callback(
    Output("prediction-output-cTree", "children"),
    Input("generate-button", "n_clicks"),
    State("max_depth", "value"),
    State("min_samples_split", "value"),
    State("min_samples_leaf", "value"),
    State("target-column-dropdown-cTree","value"),
    State("feature-columns-dropdown-cTree","value"),
)
def generate_model(n_clicks,max_depth,min_samples_split,min_samples_leaf,target_column, feature_columns):
    if(target_column != None and  any(feature_columns) and n_clicks>0):
        data = df_transformer.get_df()
        # Separate the features and the target variable
        X = np.array(data[feature_columns])
        Y = np.array(data[[target_column]])
        # if len(X) != len(Y):
        #     return html.Div([
        #     html.Div(X.shape),
        #     html.Div(Y.shape),
        #     html.Div(create_data_table(data))    
        #     ])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                            test_size = 0.2, 
                                                            random_state = 0, 
                                                            shuffle = True)
        # if len(X) == len(Y):
        #     return html.Div([
        #     html.Div(X_train.shape),
        #     html.Div(Y_train.shape),
        #     html.Div(X_test.shape),
        #     html.Div(Y_test.shape),
        #     html.Div(create_data_table(data))    
        #     ])
        samples_leaf = min_samples_leaf if min_samples_leaf != None else 1 
        samples_split= min_samples_split if min_samples_split != None else 2 
        depth=max_depth
            
        # Initialize the Decision Tree Classifier
        classifier = DecisionTreeClassifier( min_samples_leaf=samples_leaf,min_samples_split=samples_split, max_depth=depth,random_state=0)
        df_transformer.set_predictor(classifier)
        # Fit the model
        classifier.fit(X_train, Y_train)
        
        # Perform prediction
        Y_pred = classifier.predict(X_test)
        
        # Calculate evaluation metrics
        # mse = mean_squared_error(Y_test, Y_pred)
        # mae = mean_absolute_error(Y_test, Y_pred)
        # r2 = r2_score(Y_test, Y_pred)
        
        reporte = export_text(classifier, feature_names=feature_columns)
        
        ImportanciaMod1 = pd.DataFrame({'Variable': list(data[feature_columns]),
                                'Importancia': classifier.feature_importances_}).sort_values('Importancia', ascending=False)
        
        return html.Div([
            dbc.Alert(f"Criterio: {classifier.criterion}"),
            dbc.Alert(f"Importancia de variables: {classifier.feature_importances_}"),
            dbc.Alert(f"Exactitud: {accuracy_score(Y_test,Y_pred)}"),
            html.H6("Reporte de Clasificación"),
            html.Pre(classification_report(Y_test, Y_pred)),
            html.Div([html.Pre(reporte)],
                     style={'height': '20em', 'overflowY': 'scroll', 'border': '1px solid', 'padding': '10px'},
                     ),
            html.H5("Eficiencia y conformación del modelo"),
            html.Div(create_data_table(ImportanciaMod1)),
            html.H3("Realizar clasificación"),
            html.Div(id="feature-inputs-div-cTree",
                    style={
                    'marginLeft': 'auto',
                    'marginRight': 'auto',
                }),
            html.Button(
                "Predecir", 
                id="predict-button", 
                n_clicks=0,
                className="btn btn-success",
                style={
                    'marginTop' : '10px',
                    'marginLeft': '75%',
                }
            ),
            html.Div(id="manual_prediction-output-cTree")
            ],
            style={
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'width': '90%',
                'padding':10,
            } 
        )
        
        
        

@callback(
    Output("feature-inputs-div-cTree", "children"),
    Input("feature-columns-dropdown-cTree", "value"),
    Input("target-column-dropdown-cTree", "value")
)
def create_inputs(inputs, target):
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
                'width': '100%'
                
                }
        )

@callback(
    Output("manual_prediction-output-cTree", "children"),
    Input("predict-button", "n_clicks"),
    State("feature-columns-dropdown-cTree", "value"),
    State({"type": "feature-input-value", "index":ALL}, "value")
)
def make_prediction(n_clicks, feature_columns, feature_values ):
    if n_clicks > 0:
        # Read the data from a CSV file (assuming it's named "data.csv")
        regresor = df_transformer.get_preditor()
        # Create a DataFrame with the input values
        input_data = {}
        for i, col in enumerate(feature_columns):
            input_value = float(feature_values[i])
            input_data[col] = [input_value]
        input_data = pd.DataFrame(input_data)
        
        # Perform prediction on the input values
        prediction = regresor.predict(input_data.values)
        
        return dbc.Alert(f"Clasificación: {prediction[0]}")
    
    return ""