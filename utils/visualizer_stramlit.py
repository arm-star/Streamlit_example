import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
#import plotly.figure_factory as ff



def plotly_go_figure(xx, yy,the_date_column, target_column):
    """
    Interactive plotly plot
    :param xx:
    :param yy:
    :param the_date_column:
    :param target_column:
    :return:
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xx,
        y=yy,
        name="Name of Trace 1"       # this sets its legend entry
    ))

    fig.update_layout(
        title="Plot: Target Column vs Time",
        xaxis_title=the_date_column,
        yaxis_title=target_column,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    return fig

def plotly_go_two_figure(xx, yy, yy2, x_axis_name, y_axis_name, legend1, legend2, title_name):
    """

    :param xx:
    :param yy:
    :param yy2:
    :param x_axis_name:
    :param y_axis_name:
    :param legend1:
    :param legend2:
    :param title_name:
    :return:
    """


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xx,
        y=yy,
        name=legend1      # this sets its legend entry
    ))

    fig.add_trace(go.Scatter(
        x=xx,
        y=yy2,
        name=legend2       # this sets its legend entry
    ))

    fig.update_layout(
        title=title_name,
        xaxis_title=x_axis_name,
        yaxis_title=y_axis_name,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    return fig

def vis_cor_heatmap(df, dpi_res=120):
    """
    Correlation Heatmap Visualization
    :param df:
    :return:
    """
    fig1, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=dpi_res)
    hm = sns.heatmap(df.corr(), ax=ax, cmap="coolwarm", annot=True, fmt='.2f', linewidths=.05)
    fig1.suptitle('df Correlation Heatmap', fontsize=10, fontweight='bold')
    #plt.show()


def vis_group_by(df, dpi_res=120):
    fig2, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=dpi_res)
    df.groupby([pd.Grouper(freq='D')])['Open'].sum().plot()

