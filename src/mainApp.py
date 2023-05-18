from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.LUX])
# navbar = dbc.NavbarSimple(
#     id="navbar",
#     brand_href="/",
#     children=[
#         # Elemento de navegación nav-link para estilo y mx-3 para un margen de 0.75rem en ambos lados.
#         dbc.NavItem(
#             dcc.Link(
#                 "Inicio",
#                 href="/",
#                 className="nav-link mx-3",
#                 style={"whiteSpace": "nowrap"},
#             )
#         ),
#         dbc.NavItem(
#             dcc.Link(
#                 "Análisis Exploratorio de Datos (EDA)",
#                 href="/eda",
#                 className="nav-link mx-3",
#                 style={"whiteSpace": "nowrap"},
#             )
#         ),
#         dbc.NavItem(
#             dcc.Link(
#                 "Análisis de Componentes Principales (PCA)",
#                 href="/pca",
#                 className="nav-link mx-3",
#                 style={"whiteSpace": "nowrap"},
#             )
#         ),
#         dbc.NavItem(className="ml-auto"),
#     ],
#     color="red",
#     dark=False,
#     sticky="top",
# )

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("EDA", href="#"), className="example"),
        dbc.NavItem(dbc.NavLink("PCA", href="#")),        
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Clasificación", header=True),
                dbc.DropdownMenuItem("Árbol", href="#"),
                dbc.DropdownMenuItem("Bosque", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="Clasificación",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Pronóstico", header=True),
                dbc.DropdownMenuItem("Árbol", href="#"),
                dbc.DropdownMenuItem("Bosque", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="Pronóstico",
        ),
    ],
    brand="MineWizard",
    brand_href="#",
    color="primary",
    dark=True,
)

app.layout = html.Div([dcc.Location(id = "url"),
    navbar,
    html.Div(
        children=html.Div([
            html.H1('Overview'),
            html.Div('''

                This is an example of a simple Dash app with
                local, customized CSS.
            '''    ,className="example"    )

        ] )
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)