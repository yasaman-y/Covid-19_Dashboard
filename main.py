# =============================================================================
# Author            : Yasaman Yektaeian
# Email             : yasamanyektaeian@gmail.com
# Created Date      : 2021-08-22 
# Last Modified Date: 2021-08-24
# Last Modified By  : -
# Version           : 1.0.0
# =============================================================================




# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from logging import PlaceHolder
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
# import matplotlib.pylab as plty
from sklearn.metrics import  mean_absolute_error


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#------------------------------------------------------------------------------------------
#------data----------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

df_1 = pd.read_csv("Data/confirm-death-recover-active_data_by_country.csv")
df_2 = pd.read_csv("Data/confirm-death-recover-active_data_by_date.csv")
df_complete_world = pd.read_csv("Data/confirm-death-recover-active_data_daily.csv")
# df_4 = pd.read_csv("Data/coutry_all-day-row_confirm.csv")
# df_5 = pd.read_csv("Data/coutry_all-day-row_death.csv")
# df_6 = pd.read_csv("Data/coutry_all-day-row_recoveres.csv")
# df_7 = pd.read_csv("Data/daily_confirm.csv")
# df_8 = pd.read_csv("Data/daily_death.csv")
# df_9 = pd.read_csv("Data/daily_recovered.csv")
df_normal  = pd.read_csv("Data/normal_by_population.csv")
country_info = pd.read_csv("Data/country_info.csv")
country_list = country_info["Country/Region"].unique().tolist()


total_confirm = df_1["confirm_total"].sum(axis=0)
total_death = df_1["death_total"].sum(axis=0)
total_recovery = df_1["recovered_total"].sum(axis=0)
death_rate = np.round((df_1["death_total"].sum(axis=0) / df_1["confirm_total"].sum(axis=0)) * 100, 2)

# print(total_recovery)
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
fig_1 = go.Figure()
fig_1.add_trace(
    go.Scatter(
        x=df_complete_world['date'].tolist(), 
        y= df_complete_world["confirm"].tolist(), 
        mode = 'lines',
        name = 'Confirm',
        line=dict(color='orange')
        ))
fig_1.add_trace(
    go.Scatter(
        x=df_complete_world['date'].tolist(), 
        y= df_complete_world["death"].tolist(), 
        mode = 'lines',
        name = 'Death',
        line=dict(color='red')))
fig_1.add_trace(
    go.Scatter(
        x=df_complete_world['date'].tolist(), 
        y= df_complete_world["recovered"].tolist(), 
        mode = 'lines',
        name = 'Recovered',
        line=dict(color='green')))
# fig_1.add_trace(
#     go.Scatter(
#         x=df_complete_world['date'].tolist(), 
#         y= df_complete_world["active"].tolist(), 
#         mode = 'lines',
#         name = 'Active',
#         line=dict(color='blue')))

fig_1.update_layout(
    showlegend= True,
    # width=width,
    height=450,
    # hovermode='closest',
    title = {
        'text' : 'World covid Information' ,
        'xanchor' : 'center',
        'yanchor' : 'top',
        'y':0.9,
        'x':0.45,
            },
    titlefont ={
        'color' : 'black',
        'size' : 30,
        },
    font=dict( family="Courier New, monospace", size=12,),
    xaxis = dict(
        tickmode = 'array',
        tickformat="<b>%Y<b>-%m-%d",
        gridwidth=0.1,
        tickangle = 20),
    legend = {
            'bgcolor' : '#cfcfcf',  
        },
        plot_bgcolor = '#e7e7e7',
        paper_bgcolor = '#e7e7e7',
    )
#-----------------------------------------------------------------------------------------
# df_normal.sort_values(columns = ["normal"], inplacce = True)
fig_2 = go.Figure(
    data=[
        go.Bar(
            name="Death",
            x=df_normal["Country/Region"].tolist(),
            y= df_normal["normal"].tolist(),
            marker=dict(color='red'))
        ])
fig_2.update_layout(
    height = 500,
    barmode='stack',
    plot_bgcolor = '#e7e7e7',
    paper_bgcolor = '#e7e7e7',
    hovermode = 'closest',
    title = {
        'text' : 'Compare Death Related to the Population of Countries in the World',
        'xanchor' : 'center',
        'yanchor' : 'top',
        'x':0.45,
        'y':0.95
            },            
    titlefont ={
        'color' : 'black',
        'size' : 30},
    legend = {
        # 'orientation': 'H',
        'bgcolor' : '#cfcfcf',
        'xanchor' : 'right',
        'yanchor' : 'top',
        # 'x' :1,
        'y' :1.5
    },
    font = dict(
        family="sans-serif",
        size = 12,
        color='black'
    ),
    xaxis = dict(
        # tickmode = 'auto',
        ticktext = df_normal["Country/Region"].tolist(),
        # tickformat="<b>%Y<b>-%m-%d",
        tickangle = 90
    ),
    )
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
#------data----------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------


app.layout = html.Div([
                html.H1( children='Covid-19',
                            style={
                                    'text-align' : 'center',
                                    'color'  : 'black',
                                    'font-weight': 'bold', }
                                ),
                html.H4( children='Dashboard for Analysis Covid-19 Data',
                            style={
                                    'text-align' : 'center',
                                    'color'  : 'black',
                                     }
                                ),
                html.H6( children='Data resource:  COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University',
                            style={
                                    'text-align' : 'center',
                                    'color'  : 'black',
                                     }
                                ),
                # html.H6('https://github.com/CSSEGISandData/COVID-19',
                #      style={
                #                     'text-align' : 'center',
                #                     'color'  : 'black',
                #                      }),
                html.H6( children='Last update data :2021-08-23 00:00:00',
                            style={
                                    'text-align' : 'center',
                                    'color'  : 'black',
                                     }
                                ),
                html.Br(),
                html.Div([
                    dcc.Tabs([
                        dcc.Tab(label='General Information',value = 'tab_1',
                                style={'fontsize':50,
                                      'background-color':'#6879af',
                                      'color': 'white',
                                      'border-style': 'rounded',
                                    #   'border-width': '4px',
                                    #   'border-color': 'black'
                                    #   'padding': '0.625rem 2rem',
                                      'position': 'relative',
                                      'border-bottom-left-radius': '15px',
                                      'border-bottom-right-radius': '15px',
                                      'border-top-left-radius': '15px',
                                      'border-top-right-radius': '15px',
                                     },
                                children=[
                                    html.Div([
                                            html.Div([
                                                html.H4(children='Total Confirm',
                                                    style = {
                                                        'text-align' : 'center',
                                                        'color'  : '1f2c56'}
                                                ),
                                                html.P( total_confirm,
                                                    style={
                                                        'text-align' : 'center',
                                                        'color'  : 'orange',
                                                        'font-weight':'bold'}
                                                    )
                                                ], className= "three columns"),
                                            html.Div([
                                                html.H4(children='Total Death',
                                                    style = {
                                                        'text-align' : 'center',
                                                        'color'  : 'black'}
                                                ),
                                                html.P(total_death,
                                                    style={
                                                        'text-align' : 'center',
                                                        'color'  : 'red',
                                                        'font-weight':'bold'}
                                                    )
                                                ], className= "three columns"),
                                            html.Div([
                                                html.H4(children='Total Recoverd',
                                                    style = {
                                                        'text-align' : 'center',    
                                                        'color'  : 'black'}
                                                ),
                                                html.P(total_recovery,
                                                    style={
                                                        'text-align' : 'center',
                                                        'color'  : 'green',
                                                        'font-weight':'bold'}
                                                    )
                                                ], className= "three columns"),
                                            html.Div([
                                                html.H4(children='Death Rate',
                                                    style = {
                                                        'text-align' : 'center',
                                                        'color'  : 'black'}
                                                ),
                                                html.P( str(death_rate) + '%',
                                                    style={
                                                        'text-align' : 'center',
                                                        'color'  : 'blue',
                                                        'font-weight':'bold' }
                                                    )
                                                ], className= "three columns"),
                                            ], className='card_container row'),
                                # -------------------------------------------------------------

                                        # html.Hr(),
                                # -------------------------------------------------------------
                                    html.Div([
                                        html.Div([
                                            dcc.Graph(
                                            id='map',
                                            config ={'displayModeBar':False},
                                            className = 'dcc_compon  six columns',
                                            ),
                                        ]),
                                        html.Div([
                                            dcc.Graph(
                                                id='bar-graph-world',
                                                config ={'displayModeBar':False},
                                                figure = fig_1 ,
                                                className = 'dcc_compon',
                                                ),
                                        ], className = 'six columns'),
                                    ], className = 'row'),
                                # -------------------------------------------------------------
                                html.Br(),
                                html.Hr(),
                                html.H2("Data of Countries",  style={'font-weight':'bold'}),
                                        html.Div([
                                            html.Div([
                                                html.P('Select Country:', className='fixed_label', style={'color' : 'white'}),
                                                dcc.Dropdown(id='w_country',
                                                            multi = False,
                                                            clearable = True,
                                                            placeholder = 'Select Country',
                                                            options= [{'label' : c, 'value': c} for c in country_list],
                                                            className = 'dcc_compon'
                                                                ),
                                                html.Br(),
                                                    html.Div([            
                                                        html.Div([
                                                            dcc.Graph(
                                                                id='pie_chart',
                                                                config ={'displayModeBar':True},
                                                                className = 'dcc_compon',
                                                                ),
                                                        ],className = 'four columns'),
                                                        html.Div([
                                                            dcc.Graph(
                                                                id='bar_chart',
                                                                config ={'displayModeBar':True},
                                                                className = 'dcc_compon',
                                                                ),
                                                        ], className = 'eight columns'),
                                                    ],  className = 'row'),
                                                    html.Br(),
                                                    html.Hr(),
                                                    html.H5("Decomposition of Daily Data", style={'font-weight':'bold'}),
                                                    html.Br(),
                                                    html.Div([
                                                        dcc.Dropdown(
                                                            id='type',
                                                            multi = False,
                                                            clearable = True,
                                                            placeholder = 'Select data',
                                                            options= [{'label' : c, 'value': c} for c in ["confirm", "death", "recovered"]],
                                                            className = 'dcc_compon'
                                                                ),
                                                        html.Br(),
                                                        dcc.Graph(
                                                            id='time_seris_chart_trend',
                                                            config ={'displayModeBar':True},
                                                            className = 'dcc_compon',
                                                    ),
                                                         dcc.Graph(
                                                            id='time_seris_chart_seasonal',
                                                            config ={'displayModeBar':True},
                                                            className = 'dcc_compon',
                                                    ),
                                                    ]),
                                                    html.Br(),
                                                    html.H5("Rolling Move Average", style={'font-weight':'bold'}),
                                                    html.Br(),
                                                    html.Div([
                                                     dcc.Graph(
                                                            id='rolling_move',
                                                            config ={'displayModeBar':True},
                                                            className = 'dcc_compon',
                                                     )
                                                    ]),

                                            ], className= 'create_container ', id= 'cross-filter-options')
                                        ]),
                                    html.Br(),
                                    html.Br(),
                                    ]),
            

                        dcc.Tab(label='Death Information',value = 'tab_2',
                                style={
                                    'fontsize':50,
                                    'background-color':'#6879af',
                                    'color': 'white',
                                    'position': 'relative',
                                    'border-bottom-left-radius': '15px',
                                    'border-bottom-right-radius': '15px',
                                    'border-top-left-radius': '15px',
                                    'border-top-right-radius': '15px',                                   
                                },
                                 children=[
                                     html.Div([
                                        html.Div([
                                            dcc.Graph(
                                            id='map_2',
                                            config ={'displayModeBar':False},
                                            className = 'dcc_compon',
                                            ),
                                        ]),
                                        ]),
                                    html.Div([
                                         html.Div([
                                            dcc.Graph(
                                                    id='bar-graph-normal',
                                                    config ={'displayModeBar':True},
                                                    figure = fig_2,
                                                    )
                                            ]),
                                        html.Hr(),
                                        html.Br(),
                                        html.H4("Information of death in countries", style={'font-weight':'bold'}),
                                        html.Br(),
                                        html.Div([
                                            html.P('Select Country:', className='fixed_label', style={'color' : 'white'}),
                                            dcc.Dropdown(id='w_country_2',
                                                        multi = False,
                                                        clearable = True,
                                                        placeholder = 'Select Country',
                                                        options= [{'label' : c, 'value': c} for c in country_list],
                                                        className = 'dcc_compon'
                                                            ),
                                            html.Br(),
                                                html.Div([
                                                    dcc.Graph(
                                                        id='linear_zoom_death_chart',
                                                        config ={'displayModeBar':True},
                                                        className = 'dcc_compon',
                                                        ),
                                                ],className='row'),                                           
                                        ]),
                                        html.Br(),
                                        
                                    ])
                            ]),
                        
                        ]),
                    ]),
            ])


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------



@app.callback(
    Output('pie_chart', 'figure'),
    [Input('w_country', 'value')])
def update_pie(w_country):
    # print(w_country)
    colors = ['orange', 'red', 'green', 'blue']

    if w_country==None:
        w_country = 'Iran'
      


    df = df_1[df_1["Country/Region"] == str(w_country)]


    fig = go.Figure(
        data=[
            go.Pie(
                labels=['Confirm','Death','Recoverd','Active'],
                values=[df["confirm_total"].tolist()[0], df["death_total"].tolist()[0],df["recovered_total"].tolist()[0],df["active"].tolist()[0]],
                marker = dict(colors = colors),
                hoverinfo = 'label+value+percent',
                textinfo = 'label+percent',
                textfont = dict(size=13),
                hole = 0.7,
                rotation= 45
                )
            ])
    fig.update_traces(
        hoverinfo='label+value+percent', 
        textinfo='label+percent', 
        textfont_size=18,
        marker=dict(colors=colors, line=dict(color='#1f2c56', width=2)),
        )

    fig.update_layout(
        # width = 400,
        height = 300,
        plot_bgcolor = '#e7e7e7',
        paper_bgcolor = '#e7e7e7',
        hovermode = 'closest',
        title = {
            'text' : 'Total Confirm: ' + str(w_country)  ,
            'xanchor' : 'center',
            'yanchor' : 'top',
            'x':0.45,
            'y':0.95
            },
        titlefont ={
            'color' : 'black',
            'size' : 20,},
        legend = {
            # 'orientation': 'h',
            'bgcolor' : '#cfcfcf',
            'xanchor' : 'right',
            'yanchor' : 'top',
            'x' : 2,
            'y' :1
        },
        font = dict(
            family="sans-serif",
            size = 12,
            color='black'
        ),
    )   
    return fig


@app.callback(
    Output("bar_chart", 'figure'),
    [Input('w_country', 'value')])
def update_line(w_country):
    if w_country == None:
        w_country = 'Iran'
    
    df = df_2[df_2["Country/Region"] == w_country]
    df = df[df['date'] > "2021-01-01" ]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Death",
                x=df["date"].unique().tolist(),
                y=df["death"].tolist(),
                marker=dict(color='red'))
            ])
    # fig.add_trace(
    #     go.Bar(
    #         name="Recovered",
    #         x = df["date"].unique().tolist(), 
    #         y = df["recovered"].tolist() ,
    #         marker=dict(color='blue') ))

    fig.update_layout(
        height = 300,
        barmode='stack',
        plot_bgcolor = '#e7e7e7',
        paper_bgcolor = '#e7e7e7',
        hovermode = 'closest',
        title = {
            'text' : 'Death Cases Daily:  '  + str(w_country) ,
            'xanchor' : 'center',
            'yanchor' : 'top',
            'x':0.45,
            'y':0.95
                },            
        titlefont ={
            'color' : 'black',
            'size' : 20,},
        legend = {
            # 'orientation': 'H',
            'bgcolor' : '#cfcfcf',
            'xanchor' : 'right',
            'yanchor' : 'top',
            # 'x' :1,
            'y' :1.5
        },
        font = dict(
            family="sans-serif",
            size = 12,
            color='black'
        ),
        xaxis = dict(
            # tickmode = 'auto',
            ticktext = df["date"].tolist(),
            tickformat="<b>%Y<b>-%m-%d",
            tickangle = 20
        ),
        )

    return fig


@app.callback(
    Output("time_seris_chart_trend", 'figure'),
    [Input('w_country', 'value'),
    Input('type', 'value')])
def time_series_trend(w_country, type):
    
    if w_country == None:
        w_country = 'Iran'
    if type == None:
        type = 'confirm'
    
    df = df_2[df_2["Country/Region"] == w_country]

    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import STL
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.ar_model import AR
    result_STL = STL(df[str(type)],period = 12).fit()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_complete_world['date'].tolist(), 
            y= result_STL.trend.tolist(), 
            mode = 'lines',
            name = 'Confirm',
            line=dict(color='orange')
            ))


    fig.update_layout(
        showlegend= True,
        # width=width,
        height=250,
        # hovermode='closest',
        title = {
            'text' : 'Trend' ,
            'xanchor' : 'center',
            'yanchor' : 'top',
            'y':0.9,
            'x':0.45,
                },
        titlefont ={
            'color' : 'black',
            'size' : 20,
            },
        font=dict( family="Courier New, monospace", size=12,),
        xaxis = dict(
            tickmode = 'array',
            tickformat="<b>%Y<b>-%m-%d",
            gridwidth=0.1,
            tickangle = 20),
        legend = {
                'bgcolor' : '#cfcfcf',  
            },
            plot_bgcolor = '#e7e7e7',
            paper_bgcolor = '#e7e7e7',
        )
    return fig


@app.callback(
    Output("time_seris_chart_seasonal", 'figure'),
    [Input('w_country', 'value'),
    Input('type', 'value')])
def time_series_seasonal(w_country, type):
    
    if w_country == None:
        w_country = 'Iran'
    if type == None:
        type = 'confirm'
    
    df = df_2[df_2["Country/Region"] == w_country]

    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import STL
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.ar_model import AR
    result_STL = STL(df[str(type)],period = 12).fit()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_complete_world['date'].tolist(), 
            y= result_STL.seasonal.tolist(), 
            mode = 'lines',
            name = 'Confirm',
            line=dict(color='orange')
            ))


    fig.update_layout(
        showlegend= True,
        # width=width,
        height=250,
        # hovermode='closest',
        title = {
            'text' : 'Seasonality' ,
            'xanchor' : 'center',
            'yanchor' : 'top',
            'y':0.9,
            'x':0.45,
                },
        titlefont ={
            'color' : 'black',
            'size' : 20,
            },
        font=dict( family="Courier New, monospace", size=12,),
        xaxis = dict(
            tickmode = 'array',
            tickformat="<b>%Y<b>-%m-%d",
            gridwidth=0.1,
            tickangle = 20),
        legend = {
                'bgcolor' : '#cfcfcf',  
            },
            plot_bgcolor = '#e7e7e7',
            paper_bgcolor = '#e7e7e7',
        )
    return fig


@app.callback(
    Output("rolling_move", 'figure'),
    [Input('w_country', 'value'),
    Input('type', 'value')])
def rolling(w_country, type):
    
    if w_country == None:
        w_country = 'Iran'
    if type == None:
        type = 'confirm'
    
    df = df_2[df_2["Country/Region"] == w_country]

    '''will run the moving average model on a specified
    time window and it will plot the result smoothed curve'''
    scale=1.96
    plot_interval = True
    window = 20
    series = df[type]
    rolling_mean = series.rolling(window=window).mean()
    if plot_interval:
        mae = mean_absolute_error(series[window:],rolling_mean[window:])
        deviation = np.std(series[window:]-rolling_mean[window:])
        lower_bond =  rolling_mean - (mae + scale* deviation)
        upper_bond =  rolling_mean + (mae+scale*deviation)
    #   plty.plot(upper_bond, 'r--', label='Upper bound/Lower bound')
    #   plty.plot(lower_bond, 'r--')
    # plty.plot(rolling_mean, 'g', label = 'Rolling mean trend')
    # plty.plot(series[window:], label = 'Actual values')
    

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x= df['date'].tolist(), 
            y=  rolling_mean, 
            mode = 'lines',
            name ='Rolling mean trend',
            line=dict(color='green')
            ))

    fig.add_trace(
        go.Scatter(
            x= df['date'].tolist(), 
            y=  series[window:], 
            mode = 'lines',
            name = 'Actual values',
            line=dict(color='blue')
            ))

    fig.add_trace(
        go.Scatter(
            x= df['date'].tolist(), 
            y=  upper_bond, 
            mode = 'lines',
            name = 'Upper bound',
            line=dict(color='red', dash='dash')
            ))
    fig.add_trace(
        go.Scatter(
            x= df['date'].tolist(), 
            y=  lower_bond, 
            mode = 'lines',
            name = 'Lower bound',
            line=dict(color='red',  dash='dash')
            ))
    
      
    # fig.update_xaxes(
    #             rangeslider_visible = True, 
    #             rangeselector= dict(
    #                                 buttons=list([
    #                                     dict(count=1, label='1y', step='year' , stepmode="backward"),
    #                                     dict(count=2, label='3y', step='year' , stepmode="backward" ),
    #                                     dict(count=3, label='5y', step='year' , stepmode="backward" ),
    #                                     dict(step='all' )
    #                                 ])
    #                             )
    #             )
    
    fig.update_layout(
        showlegend= True,
        # width=width,
        height=400,
        # hovermode='closest',
        title = {
            'text' : 'Moving average(window size = {})'.format(window) ,
            'xanchor' : 'center',
            'yanchor' : 'top',
            'y':0.9,
            'x':0.45,
                },
        titlefont ={
            'color' : 'black',
            'size' : 20,
            },
        font=dict( family="Courier New, monospace", size=12,),
        xaxis = dict(
            tickmode = 'array',
            tickformat="<b>%Y<b>-%m-%d",
            gridwidth=0.1,
            tickangle = 20),
        legend = {
                'bgcolor' : '#cfcfcf',  
            },
            plot_bgcolor = '#e7e7e7',
            paper_bgcolor = '#e7e7e7',
        )
    return fig


@app.callback(
    Output("map", 'figure'),
    [Input('w_country', 'value')])
def map_info(w_country):

    df_map = pd.merge(left = country_info, right= df_1, on="Country/Region")
    fig = px.scatter_geo(df_map, 
                        locations="iso_alpha", 
                        color="Country/Region",
                        hover_name= 'Country/Region',     #"country", 
                        size="confirm_total",
                        projection="natural earth")
    fig.update_layout(
        hovermode='closest',
        title = {
            'text' : 'Total Confirm' ,
            'xanchor' : 'center',
            'yanchor' : 'top',
            'y':0.95,
            'x':0.45,
             },
        titlefont ={
            'color' : 'black',
            'size' : 30,
            },
        font = dict(
            family="sans-serif",
            size = 12,
            color='black'
        ),
        legend = {
            'bgcolor' : '#cfcfcf',  
        },
        plot_bgcolor = '#e7e7e7',
        paper_bgcolor = '#e7e7e7',

    )
    return fig


# -----death Tab--------------------------------------------------------------------------------

@app.callback(
    Output("map_2", 'figure'),
    [Input('w_country', 'value')])
def map_info(w_country):
    df_map = pd.merge(left = country_info, right= df_1[["Country/Region", "confirm_total", "death_total","recovered_total"]], on="Country/Region")
    confirm_map = 20
    death_map = 45
    recovery_map = 56
    
    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lon = df_map["Long"].tolist(),
                lat = df_map["Lat"].tolist(),
                mode = 'markers',
                marker = go.scattermapbox.Marker(
                    size = df_map["death_total"]/900,
                    color = df_map["death_total"],
                    colorscale = 'hsv',
                    # showscale = 'false',
                    sizemode = 'area',
                    opacity = 0.4
                ),
                )
            ])

    fig.update_traces(
        hoverinfo='text', 
        hovertext = 
        '<b> Country </b>:' + df_map["Country/Region"].astype(str) + '<br>' + 
        '<b> confirm </b>: ' +  str(confirm_map)  + '</br>' +
        '<b> Death </b>: ' +  str(death_map) + '</br>' +
        '<b> recovered </b>: ' + str(recovery_map) + '</br>' 
        )

    fig.update_layout(
        hovermode='closest',
        height = 500,
        # autosize = True
        mapbox=dict(
            accesstoken='pk.eyJ1IjoieWFzYW1hbi15IiwiYSI6ImNrc29rd2loeDB4bW0yb2xleGxjNmpmNDYifQ.2xPaR35gonyk-qkZ7TaeKw',
            # margin = {"r":0, "t" :0, "l":0, "b":0},
            bearing=0,
            #     center = go.layout.mapbox.Center(lat=36, lon=5.4 ),
            # style = 'dark',
            #     zoom = 1.2
            center=go.layout.mapbox.Center(
                lat=32,
                lon=-53
            ),
            # pitch=1.2,
            zoom=1.2
        ),
        title = {
            'text' : 'Total Death' ,
            'xanchor' : 'center',
            'yanchor' : 'top',
            'y':0.95,
            'x':0.45,
             },
        titlefont ={
            'color' : 'black',
            'size' : 30,
            },
        font = dict(
            family="sans-serif",
            size = 12,
            color='white'
        ),

    )
    return fig



@app.callback(
    Output("linear_zoom_death_chart", 'figure'),
    [Input('w_country_2', 'value')])   
def death_zoom(w_country):
    if w_country == None:
        w_country = 'Iran'

    
    df = df_2[df_2["Country/Region"] == w_country]


    fig = px.line(df, x= 'date', y = 'death', title='Daily Death')
    fig.update_xaxes(
                rangeslider_visible = True, 
                rangeselector= dict(
                                    buttons=list([
                                        dict(count=1, label='1y', step='year' , stepmode="backward"),
                                        dict(count=2, label='3y', step='year' , stepmode="backward" ),
                                        # dict(count=3, label='5y', step='year' , stepmode="backward" ),
                                        dict(step='all' )
                                    ])
                                )
                )
    fig.update_layout(
        showlegend= True,
        font=dict( 
            family="Courier New, monospace",
            size=12,),
         xaxis = dict(
             tickmode = 'array',
             tickformat="<b>%Y<b>-%m-%d",
             gridwidth=0.1,
             tickangle = 20  ))

    fig.update_layout(
        showlegend= True,
        # width=width,
        height=500,
        # hovermode='closest',
        title = {
            # 'text' : 'trend' ,
            'xanchor' : 'center',
            'yanchor' : 'top',
            'y':0.9,
            'x':0.45,
                },
        titlefont ={
            'color' : 'black',
            'size' : 20,
            },
        font=dict( family="Courier New, monospace", size=12,),
        xaxis = dict(
            tickmode = 'array',
            tickformat="<b>%Y<b>-%m-%d",
            gridwidth=0.1,
            tickangle = 20),
        legend = {
                'bgcolor' : '#cfcfcf',  
            },
            # plot_bgcolor = '#e7e7e7',
            paper_bgcolor = '#e7e7e7',
        )
    return fig


#------------------------------------------------------------------------------------------



if __name__ == '__main__':
    app.run_server(debug=True)  