import numpy as np
import plotly.graph_objs as go
import streamlit as st

# Generate random linear data (x, y)
np.random.seed(20)
x = np.linspace(-5, 5, 50)
y = 0.5 * x + np.random.normal(size=x.size)

def perpendicular_projection(x0, y0, slope, intercept):
    if np.isinf(slope):
        x_proj = intercept
        y_proj = y0
    else:
        x_proj = (x0 + slope * (y0 - intercept)) / (slope**2 + 1)
        y_proj = slope * x_proj + intercept
    return x_proj, y_proj

# Least squares functions...
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

# Streamlit components for interactivity
st.title("Interactive Regression Line Fitting")
slope = st.slider("Slope", -3.0, 3.0, 1.0)
intercept = st.slider("Intercept", -5.0, 5.0, 0.0)
distance_type = st.selectbox("Distance Type", ["vertical", "horizontal", "perpendicular"])

def plot_regression_plotly(slope=1.0, intercept=0.0, distance_type="vertical"):
    # Same plotting function as before...

    # Create the layout for the plot
    layout = go.Layout(
        title=f'Sum of squared distances ({distance_type}): User Line vs Least Squares',
        xaxis=dict(title='x', range=[-6, 6]),
        yaxis=dict(title='y', range=[-6, 6]),
        showlegend=True,
        width=900,  
        height=600,  
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Create the figure and display it
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)

# Update plot interactively
plot_regression_plotly(slope, intercept, distance_type)
