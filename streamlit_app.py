import numpy as np
import plotly.graph_objs as go
import streamlit as st

# Generate random linear data (x, y)
np.random.seed(20)
x = np.linspace(-5, 5, 50)
y = 0.5 * x + np.random.normal(size=x.size)

# Function to calculate perpendicular distance from a point to a line and its projection point on the line
def perpendicular_projection(x0, y0, slope, intercept):
    x_proj = (x0 + slope * (y0 - intercept)) / (slope**2 + 1)
    y_proj = slope * x_proj + intercept
    return x_proj, y_proj

# Function to plot regression line and distances interactively with Plotly
def plot_regression_plotly(slope=1.0, intercept=0.0, distance_type="vertical"):
    # Compute the fitted regression line
    y_pred = slope * x + intercept

    # Initialize traces for the plot
    data = []
    
    # Trace for the data points
    data.append(go.Scatter(x=x, y=y, mode='markers', name='Data points', marker=dict(color='black')))
    
    # Trace for the fitted regression line
    line_x = np.linspace(-6, 6, 100)
    line_y = slope * line_x + intercept
    data.append(go.Scatter(x=line_x, y=line_y, mode='lines', name=f'Fitted line: y = {slope:.2f}x + {intercept:.2f}', line=dict(color='red')))
    
    # Add residual lines and calculate SSD
    ssd = 0
    for i in range(len(x)):
        if distance_type == "vertical":
            # Vertical distance (difference in y)
            data.append(go.Scatter(x=[x[i], x[i]], y=[y[i], y_pred[i]], mode='lines', line=dict(color='pink', dash='dash')))
            ssd += (y[i] - y_pred[i]) ** 2
        elif distance_type == "horizontal":
            # Horizontal distance (difference in x)
            x_proj = (y[i] - intercept) / slope
            data.append(go.Scatter(x=[x[i], x_proj], y=[y[i], y[i]], mode='lines', line=dict(color='green', dash='dash')))
            ssd += (x[i] - x_proj) ** 2
        elif distance_type == "perpendicular":
            # Perpendicular distance
            x_proj, y_proj = perpendicular_projection(x[i], y[i], slope, intercept)
            data.append(go.Scatter(x=[x[i], x_proj], y=[y[i], y_proj], mode='lines', line=dict(color='blue', dash='dash')))
            perp_dist = np.sqrt((x[i] - x_proj)**2 + (y[i] - y_proj)**2)
            ssd += perp_dist ** 2
    
    # Create the layout for the plot
    layout = go.Layout(
        title=f'Sum of squared distances ({distance_type}): {ssd:.2f}',
        xaxis=dict(title='x', range=[-6, 6]),
        yaxis=dict(title='y', range=[-6, 6]),
        showlegend=True,
        width=900,
        height=600,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Create the figure and display it
    fig = go.Figure(data=data, layout=layout)
    return fig

# Streamlit UI components
st.title('Interactive Regression Plot with Streamlit')

slope = st.slider('Slope', min_value=-3.0, max_value=3.0, step=0.1, value=1.0)
intercept = st.slider('Intercept', min_value=-5.0, max_value=5.0, step=0.1, value=0.0)
distance_type = st.selectbox('Distance type', ['vertical', 'horizontal', 'perpendicular'])

# Update the plot
fig = plot_regression_plotly(slope=slope, intercept=intercept, distance_type=distance_type)

# Display the plot using Streamlit
st.plotly_chart(fig)
