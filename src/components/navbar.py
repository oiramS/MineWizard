from dash import Dash, html
import dash_bootstrap_components as dbc


def render(app: Dash) -> html.Div:
    '''
    navbar 
    '''
    return dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("INICIO", href="/")),
        dbc.NavItem(dbc.NavLink("EDA", href="/eda")),
        dbc.NavItem(dbc.NavLink("PCA", href="/pca")),        
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Árbol", href="/c_tree"),
                dbc.DropdownMenuItem("Bosque", href="/c_forest"),
            ],
            nav=True,
            in_navbar=True,
            label="Clasificación",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Árbol", href="/p_tree"),
                dbc.DropdownMenuItem("Bosque", href="/p_forest"),
            ],
            nav=True,
            in_navbar=True,
            label="Pronóstico",
        ),
    ],
    brand="MineWizard",
    brand_href="/",
    color="primary",
    dark=True,
)