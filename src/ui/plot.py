import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_growth_area(data, window_size, final):
    """
    line plot of area growth wrt time
    :param final: if final is true returns a figure else a scatter
    :param window_size: size of the window to compute the rolling mean
    :param data: data from the core algorithm
    :return: figure
    """
    x_axis_label = "Time (s)"
    y_axis_label = "Area (um2)"

    y_col = "area_growth"
    x_col = "time"

    data[y_col] = data[y_col].rolling(window_size, min_periods=1).mean()

    scatter = go.Scatter(x=data[x_col], y=data[y_col])
    if final:
        fig = go.Figure(data=scatter)
        fig.update_layout(xaxis_title=x_axis_label)
        fig.update_layout(yaxis_title=y_axis_label)
        fig.update_layout(title="Area growth over time")
        return fig
    return scatter


def plot_cytoplasm_intensity(data, window_size, final):
    """
    line plot of cytoplasm mean intensity wrt time
    :param final: if final is true returns a figure else a scatter
    :param window_size: size of the window to compute the rolling mean
    :param data: data from the core algorithm
    :return: figure
    """
    x_axis_label = "Time (s)"
    y_axis_label = "Mean intensity"

    y_col = "cytoplasm_intensity_mean"
    x_col = "time"

    data[y_col] = data[y_col].rolling(window_size, min_periods=1).mean()

    scatter = go.Scatter(x=data[x_col], y=data[y_col])
    if final:
        fig = go.Figure(data=scatter)
        fig.update_layout(xaxis_title=x_axis_label)
        fig.update_layout(yaxis_title=y_axis_label)
        fig.update_layout(title="Mean cytoplasm intensity over time")
        return fig
    return scatter


def plot_membrane_intensity(data, window_size, final):
    """
    line plot of cytoplasm mean intensity wrt time
    :param final: if final is true returns a figure else a scatter
    :param window_size: size of the window to compute the rolling mean
    :param data: data from the core algorithm
    :return: figure
    """
    x_axis_label = "Time (s)"
    y_axis_label = "Mean intensity"

    y_col = "membrane_intensity_mean"
    x_col = "time"

    data[y_col] = data[y_col].rolling(window_size, min_periods=1).mean()

    scatter = go.Scatter(x=data[x_col], y=data[y_col])
    if final:
        fig = go.Figure(data=scatter)
        fig.update_layout(xaxis_title=x_axis_label)
        fig.update_layout(yaxis_title=y_axis_label)
        fig.update_layout(title="Mean membrane intensity over time")
        return fig
    return scatter


def plot_direction_angle(data, window_size, final):
    """
    line plot of growth direction angle wrt time
    :param final: if final is true returns a figure else a scatter
    :param window_size: size of the window to compute the rolling mean
    :param data: data from the core algorithm
    :return: figure
    """
    x_axis_label = "Time (s)"
    y_axis_label = "Growth direction angle"

    y_col = "growth_direction_angle"
    x_col = "time"

    data[y_col] = data[y_col].rolling(window_size, min_periods=1).mean()

    scatter = go.Scatter(x=data[x_col], y=data[y_col])
    if final:
        fig = go.Figure(data=scatter)
        fig.update_layout(xaxis_title=x_axis_label)
        fig.update_layout(yaxis_title=y_axis_label)
        fig.update_layout(title="Growth direction angle over time")
        return fig
    return scatter


def plot_membrane_heatmap(data, membrane_data, membrane_xs, colorscale, other_fig, smooth=False):
    """
    heatmap plot of the membrane with time as xaxis, curvilinear abscissa as yaxis and intensity
    as color.
    :param other_fig: other graph to lay on this one
    :param colorscale: name of the color scale
    :param data: data from the core algorithm
    :param membrane_data: distribution of intensity along the membrane
    :param membrane_xs: curvilinear abscissa of the distribution
    :return: figure
    """
    x_axis_label = "Time (s)"
    y_axis_label = "Membrane position (um)"
    h_data = membrane_data.T
    if smooth == 'none':
        smooth = False

    full_fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig = go.Heatmap(
        x=data["time"],
        y=membrane_xs,
        z=h_data,
        colorscale=colorscale,
        zsmooth = smooth)

    full_fig.add_trace(fig)
    if other_fig is not None:
        full_fig.add_trace(other_fig, secondary_y=True)

    full_fig.update_layout(xaxis_title=x_axis_label)
    full_fig.update_layout(yaxis_title=y_axis_label)

    return full_fig
