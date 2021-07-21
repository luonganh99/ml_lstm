import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  # Deeplearing API
from keras.layers import LSTM, Dropout, Dense
import numpy as np
import lstm

app = dash.Dash()
server = app.server

microsoftCloseData = lstm.train("MSFT", 6385)
facebookCloseData = lstm.train("FB", 1104)
teslaCloseData = lstm.train("TSLA", 1485)
appleCloseData = lstm.train("AAPL", 6690)


df = pd.read_csv("./stock_data.csv")

app.layout = html.Div(
    [
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="Apple Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Actual closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual Apple Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=appleCloseData[0].index,
                                                y=appleCloseData[1]["Close"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "LSTM Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=appleCloseData[0].index,
                                                y=appleCloseData[1]["Predictions"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "Actual price rate of change",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual price rate of change Apple Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=appleCloseData[2].index,
                                                y=appleCloseData[3]["Rate"],
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "LSTM Predicted price rate of change",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted price rate of change Apple Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=appleCloseData[2].index,
                                                y=appleCloseData[3]["Predictions"],
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Tesla Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Actual closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual Tesla Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=teslaCloseData[0].index,
                                                y=teslaCloseData[1]["Close"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "LSTM Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted Tesla Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=teslaCloseData[0].index,
                                                y=teslaCloseData[1]["Predictions"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "Actual price rate of change",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual price rate of change Tesla Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=teslaCloseData[2].index,
                                                y=teslaCloseData[3]["Rate"],
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "LSTM Predicted price rate of change",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted price rate of change Tesla Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=teslaCloseData[2].index,
                                                y=teslaCloseData[3]["Predictions"],
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Microsoft Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Actual closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual Microsoft Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=microsoftCloseData[0].index,
                                                y=microsoftCloseData[1]["Close"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "LSTM Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted Microsoft Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=microsoftCloseData[0].index,
                                                y=microsoftCloseData[1]["Predictions"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "Actual price rate of change",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual price rate of change Microsoft Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=microsoftCloseData[2].index,
                                                y=microsoftCloseData[3]["Rate"],
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "LSTM Predicted price rate of change",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted price rate of change Microsoft Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=microsoftCloseData[2].index,
                                                y=microsoftCloseData[3]["Predictions"],
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Facebook Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Actual closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual Facebook Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=facebookCloseData[0].index,
                                                y=facebookCloseData[1]["Close"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "LSTM Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted Facebook Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=facebookCloseData[0].index,
                                                y=facebookCloseData[1]["Predictions"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "Actual price rate of change",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual price rate of change Facebook Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=facebookCloseData[2].index,
                                                y=facebookCloseData[3]["Rate"],
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "LSTM Predicted price rate of change",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted price rate of change Facebook Data",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=facebookCloseData[2].index,
                                                y=facebookCloseData[3]["Predictions"],
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
            ],
        ),
    ]
)


# @app.callback(Output("highlow", "figure"), [Input("my-dropdown", "value")])
# def update_graph(selected_dropdown):
#     dropdown = {
#         "TSLA": "Tesla",
#         "AAPL": "Apple",
#         "FB": "Facebook",
#         "MSFT": "Microsoft",
#     }
#     trace1 = []
#     trace2 = []
#     for stock in selected_dropdown:
#         trace1.append(
#             go.Scatter(
#                 x=df[df["Stock"] == stock]["Date"],
#                 y=df[df["Stock"] == stock]["High"],
#                 mode="lines",
#                 opacity=0.7,
#                 name=f"High {dropdown[stock]}",
#                 textposition="bottom center",
#             )
#         )
#         trace2.append(
#             go.Scatter(
#                 x=df[df["Stock"] == stock]["Date"],
#                 y=df[df["Stock"] == stock]["Low"],
#                 mode="lines",
#                 opacity=0.6,
#                 name=f"Low {dropdown[stock]}",
#                 textposition="bottom center",
#             )
#         )
#     traces = [trace1, trace2]
#     data = [val for sublist in traces for val in sublist]
#     figure = {
#         "data": data,
#         "layout": go.Layout(
#             colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
#             height=600,
#             title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
#             xaxis={
#                 "title": "Date",
#                 "rangeselector": {
#                     "buttons": list(
#                         [
#                             {
#                                 "count": 1,
#                                 "label": "1M",
#                                 "step": "month",
#                                 "stepmode": "backward",
#                             },
#                             {
#                                 "count": 6,
#                                 "label": "6M",
#                                 "step": "month",
#                                 "stepmode": "backward",
#                             },
#                             {"step": "all"},
#                         ]
#                     )
#                 },
#                 "rangeslider": {"visible": True},
#                 "type": "date",
#             },
#             yaxis={"title": "Price (USD)"},
#         ),
#     }
#     return figure


# @app.callback(Output("volume", "figure"), [Input("my-dropdown2", "value")])
# def update_graph(selected_dropdown_value):
#     dropdown = {
#         "TSLA": "Tesla",
#         "AAPL": "Apple",
#         "FB": "Facebook",
#         "MSFT": "Microsoft",
#     }
#     trace1 = []
#     for stock in selected_dropdown_value:
#         trace1.append(
#             go.Scatter(
#                 x=df[df["Stock"] == stock]["Date"],
#                 y=df[df["Stock"] == stock]["Volume"],
#                 mode="lines",
#                 opacity=0.7,
#                 name=f"Volume {dropdown[stock]}",
#                 textposition="bottom center",
#             )
#         )
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {
#         "data": data,
#         "layout": go.Layout(
#             colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
#             height=600,
#             title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
#             xaxis={
#                 "title": "Date",
#                 "rangeselector": {
#                     "buttons": list(
#                         [
#                             {
#                                 "count": 1,
#                                 "label": "1M",
#                                 "step": "month",
#                                 "stepmode": "backward",
#                             },
#                             {
#                                 "count": 6,
#                                 "label": "6M",
#                                 "step": "month",
#                                 "stepmode": "backward",
#                             },
#                             {"step": "all"},
#                         ]
#                     )
#                 },
#                 "rangeslider": {"visible": True},
#                 "type": "date",
#             },
#             yaxis={"title": "Transactions Volume"},
#         ),
#     }
#     return figure


if __name__ == "__main__":
    app.run_server(debug=True)
