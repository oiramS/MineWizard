from dash import Dash
import dash_bootstrap_components as dbc

from components.layout import create_layout

PORT = 3031

def main() -> None:
    '''
    entry point function
    '''
    app = Dash(
        external_stylesheets=[dbc.themes.LUX],
        suppress_callback_exceptions=True,
        )
    app.title = "MineWizard"
    app.layout = create_layout(app)
    app.run(
        port=PORT,
        debug=True
    )

if __name__ == '__main__':
    main()
    