from dash import Dash, html
import dash_bootstrap_components as dbc


def render(app: Dash) -> html.Div:
    '''
    navbar 
    '''
    return dbc.NavbarSimple(
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