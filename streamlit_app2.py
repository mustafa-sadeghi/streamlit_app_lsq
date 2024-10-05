import numpy as np
import plotly.graph_objs as go
import streamlit as st

# Generate random linear data (x, y)
np.random.seed(20)
x = np.linspace(-5, 5, 50)
y = 0.5 * x + np.random.normal(size=x.size)

# Function to calculate perpendicular distance from a point to a line and its projection point on the line
def perpendicular_projection(x0, y0, slope, intercept):
    if np.isinf(slope):
        x_proj = intercept
        y_proj = y0
    else:
        x_proj = (x0 + slope * (y0 - intercept)) / (slope**2 + 1)
        y_proj = slope * x_proj + intercept
    return x_proj, y_proj

# Functions to compute least squares regression lines for each distance type
def compute_least_squares_vertical(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept

def compute_least_squares_horizontal(x, y):
    c, d = np.polyfit(y, x, 1)
    if c != 0:
        slope = 1 / c
        intercept = -d / c
    else:
        slope = 0
        intercept = d
    return slope, intercept

def compute_least_squares_perpendicular(x, y):
    # Center the data
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_cent = x - x_mean
    y_cent = y - y_mean

    # Stack x and y
    data = np.vstack((x_cent, y_cent))

    # Compute the covariance matrix
    cov = np.cov(data)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Direction vector of the line
    direction = eigvecs[:, 1]

    # Compute slope and intercept
    if direction[0] != 0:
        slope = direction[1] / direction[0]
        intercept = y_mean - slope * x_mean
    else:
        slope = np.inf
        intercept = x_mean  # For vertical line x = intercept

    return slope, intercept

# Function to plot regression line and distances interactively with Plotly
def plot_regression_plotly(slope=1.0, intercept=0.0, distance_type="vertical"):
    # Compute the fitted regression line
    y_pred = slope * x + intercept

    # Compute the least squares line based on distance type
    if distance_type == "vertical":
        ls_slope, ls_intercept = compute_least_squares_vertical(x, y)
    elif distance_type == "horizontal":
        ls_slope, ls_intercept = compute_least_squares_horizontal(x, y)
    elif distance_type == "perpendicular":
        ls_slope, ls_intercept = compute_least_squares_perpendicular(x, y)

    # Initialize traces for the plot
    data = []

    # Trace for the data points
    data.append(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data points',
        marker=dict(color='black')
    ))

    # Trace for the user-defined regression line
    line_x = np.linspace(-6, 6, 100)
    if np.isinf(slope):
        # Vertical line at x = intercept
        data.append(go.Scatter(
            x=[intercept, intercept],
            y=[-6, 6],
            mode='lines',
            name=f'User Line: x = {intercept:.2f}',
            line=dict(color='red')
        ))
    else:
        line_y = slope * line_x + intercept
        data.append(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            name=f'User Line: y = {slope:.2f}x + {intercept:.2f}',
            line=dict(color='red')
        ))

    # Trace for the least squares line
    if np.isinf(ls_slope):
        # Vertical line at x = ls_intercept
        data.append(go.Scatter(
            x=[ls_intercept, ls_intercept],
            y=[-6, 6],
            mode='lines',
            name='Least Squares Line',
            line=dict(color='green')
        ))
    else:
        line_x_ls = np.linspace(-6, 6, 100)
        line_y_ls = ls_slope * line_x_ls + ls_intercept
        data.append(go.Scatter(
            x=line_x_ls,
            y=line_y_ls,
            mode='lines',
            name=f'Least Squares Line: y = {ls_slope:.2f}x + {ls_intercept:.2f}',
            line=dict(color='green')
        ))

    # Add residual lines and calculate SSD for user line
    ssd = 0
    for i in range(len(x)):
        if distance_type == "vertical":
            # Vertical distance (difference in y)
            data.append(go.Scatter(
                x=[x[i], x[i]],
                y=[y[i], y_pred[i]],
                mode='lines',
                line=dict(color='pink', dash='dash'),
                showlegend=False
            ))
            ssd += (y[i] - y_pred[i]) ** 2
        elif distance_type == "horizontal":
            # Horizontal distance (difference in x)
            if slope != 0 and not np.isinf(slope):
                x_proj = (y[i] - intercept) / slope
                data.append(go.Scatter(
                    x=[x[i], x_proj],
                    y=[y[i], y[i]],
                    mode='lines',
                    line=dict(color='pink', dash='dash'),
                    showlegend=False
                ))
                ssd += (x[i] - x_proj) ** 2
            else:
                # Horizontal line, x_proj is undefined
                data.append(go.Scatter(
                    x=[x[i], np.mean(x)],
                    y=[y[i], y[i]],
                    mode='lines',
                    line=dict(color='pink', dash='dash'),
                    showlegend=False
                ))
                ssd += (x[i] - np.mean(x)) ** 2  # Approximate
        elif distance_type == "perpendicular":
            # Perpendicular distance
            x_proj, y_proj = perpendicular_projection(x[i], y[i], slope, intercept)
            data.append(go.Scatter(
                x=[x[i], x_proj],
                y=[y[i], y_proj],
                mode='lines',
                line=dict(color='pink', dash='dash'),
                showlegend=False
            ))
            perp_dist = np.sqrt((x[i] - x_proj)**2 + (y[i] - y_proj)**2)
            ssd += perp_dist ** 2

    # Calculate SSD for least squares line
    ssd_ls = 0
    for i in range(len(x)):
        if distance_type == "vertical":
            y_pred_ls = ls_slope * x[i] + ls_intercept
            ssd_ls += (y[i] - y_pred_ls) ** 2
        elif distance_type == "horizontal":
            if ls_slope != 0 and not np.isinf(ls_slope):
                x_proj_ls = (y[i] - ls_intercept) / ls_slope
                ssd_ls += (x[i] - x_proj_ls) ** 2
            else:
                ssd_ls += (x[i] - np.mean(x)) ** 2  # Approximate
        elif distance_type == "perpendicular":
            x_proj_ls, y_proj_ls = perpendicular_projection(x[i], y[i], ls_slope, ls_intercept)
            perp_dist_ls = np.sqrt((x[i] - x_proj_ls)**2 + (y[i] - y_proj_ls)**2)
            ssd_ls += perp_dist_ls ** 2

    # Create the layout for the plot with larger size
    layout = go.Layout(
        title=f'Sum of squared distances ({distance_type}):<br>User Line SSD = {ssd:.2f}, Least Squares SSD = {ssd_ls:.2f}',
        xaxis=dict(title='x', range=[-6, 6]),
        yaxis=dict(title='y', range=[-6, 6]),
        showlegend=True,
        width=900,
        height=600,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Create the figure and return it
    fig = go.Figure(data=data, layout=layout)
    return fig

# Streamlit interface for interactivity
st.title("Interactive Linear Regression Visualization")

# Create sliders and dropdown using Streamlit
slope = st.slider("Slope (m)", min_value=-3.0, max_value=3.0, step=0.1, value=1.0)
intercept = st.slider("Intercept (b)", min_value=-5.0, max_value=5.0, step=0.1, value=0.0)
distance_type = st.selectbox("Distance Type", ["vertical", "horizontal", "perpendicular"])

# Call the plotting function with user inputs
fig = plot_regression_plotly(slope, intercept, distance_type)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
