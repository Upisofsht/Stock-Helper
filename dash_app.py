import dash
from dash import dcc, html

# 初始化
app = dash.Dash(__name__)

# --- 關鍵：這行一定要加，部署才會成功 ---
server = app.server 
# -------------------------------------

app.layout = html.Div([
    html.H1("我的 Dash 手機版 App")
])

if __name__ == '__main__':
    app.run_server(debug=True)